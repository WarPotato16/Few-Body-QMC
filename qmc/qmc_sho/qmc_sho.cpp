#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "harmonic_potential.hpp"

using Mask = unsigned int;
using Bond = std::pair<int, int>;

const double negativeInfinity = -std::numeric_limits<double>::infinity();

// Model and operator types
enum class ParticleModel {
    HardcoreBoson,
    SpinlessFermion
};

enum class BoundaryCondition {
    Periodic,
    Open
};

enum class OperatorKind {
    Identity,
    Diagonal,
    Hopping,
    Potential
};

struct BondOperator {
    OperatorKind kind = OperatorKind::Identity;

    // Bond index for Diagonal/Hopping, site index for Potential.
    int bond = -1;
};

struct QmcParameters {
    int latticeSize = 4;
    int particleCount = 2;
    double beta = 1.0;
    double hoppingT = 1.0;
    int stringLength = 200;
    long long thermalizationSweeps = 1000;
    long long measurementSweeps = 10000;
    int measureEvery = 1;
    int blockCount = 50;
    ParticleModel model = ParticleModel::HardcoreBoson;
    BoundaryCondition boundary = BoundaryCondition::Periodic;
    unsigned int seed = 12345;
    bool showProgress = true;
    sho::Parameters harmonicTrap;

    // Used only by parallel runs so the terminal can show per-chain progress.
    std::atomic<long long>* sharedChainProgress = nullptr;
    int chainIndex = -1;
    int chainCount = 0;
};

struct AcceptanceCounter {
    long long attempts = 0;
    long long accepted = 0;
};

struct LocalMatrixElement {
    bool valid = false;
    Mask newState = 0;
    double absWeight = 0.0;
    int sign = 1;
};

struct SignedLogWeight {
    bool valid = false;
    double logAbsWeight = negativeInfinity;
    int sign = 1;
    int activeCount = 0;
};

struct MeasurementBlock {
    long long count = 0;
    double activeSum = 0.0;
    double diagonalSum = 0.0;
    double hoppingSum = 0.0;
    double potentialOperatorSum = 0.0;
    double signSum = 0.0;
    double signedActiveSum = 0.0;
    double signedPotentialEnergySum = 0.0;
};

struct QmcResult {
    int latticeSize = 0;
    int particleCount = 0;
    double beta = 0.0;
    double hoppingT = 0.0;
    int stringLength = 0;
    int basisSize = 0;
    double averageActive = 0.0;
    double averageDiagonal = 0.0;
    double averageHopping = 0.0;
    double averagePotentialOperators = 0.0;
    double averagePotentialEnergy = 0.0;
    double averageSign = 1.0;
    double signError = 0.0;
    double energy = 0.0;
    double energyError = 0.0;
    double cutoffHitFraction = 0.0;
    double diagonalAcceptance = 0.0;
    double wormAcceptance = 0.0;
    double stateAcceptance = 0.0;
};

// Basic utilities
int popcountMask(Mask x) {
    return __builtin_popcount(x);
}

bool occupied(Mask state, int site) {
    return ((state >> site) & 1U) != 0U;
}

Mask flipTwoSites(Mask state, int i, int j) {
    state ^= (1U << i);
    state ^= (1U << j);
    return state;
}

std::string modelName(ParticleModel model) {
    if (model == ParticleModel::HardcoreBoson) {
        return "hardcore_boson";
    }
    return "spinless_fermion";
}

std::string boundaryName(BoundaryCondition boundary) {
    if (boundary == BoundaryCondition::Periodic) {
        return "periodic";
    }
    return "open";
}

ParticleModel parseParticleModel(const std::vector<std::string>& args, ParticleModel fallback) {
    for (int i = 0; i + 1 < static_cast<int>(args.size()); ++i) {
        if (args[i] == "--model") {
            if (args[i + 1] == "boson" || args[i + 1] == "hardcore_boson") {
                return ParticleModel::HardcoreBoson;
            }
            if (args[i + 1] == "fermion" || args[i + 1] == "spinless_fermion") {
                return ParticleModel::SpinlessFermion;
            }
            throw std::runtime_error("Unknown --model. Use boson or fermion.");
        }
    }
    return fallback;
}

BoundaryCondition parseBoundaryCondition(const std::vector<std::string>& args, BoundaryCondition fallback) {
    for (int i = 0; i + 1 < static_cast<int>(args.size()); ++i) {
        if (args[i] == "--boundary") {
            if (args[i + 1] == "periodic") {
                return BoundaryCondition::Periodic;
            }
            if (args[i + 1] == "open") {
                return BoundaryCondition::Open;
            }
            throw std::runtime_error("Unknown --boundary. Use periodic or open.");
        }
    }
    return fallback;
}

int parseIntArg(const std::vector<std::string>& args, const std::string& name, int fallback) {
    for (int i = 0; i + 1 < static_cast<int>(args.size()); ++i) {
        if (args[i] == name) {
            return std::stoi(args[i + 1]);
        }
    }
    return fallback;
}

long long parseLongLongArg(const std::vector<std::string>& args, const std::string& name, long long fallback) {
    for (int i = 0; i + 1 < static_cast<int>(args.size()); ++i) {
        if (args[i] == name) {
            return std::stoll(args[i + 1]);
        }
    }
    return fallback;
}

double parseDoubleArg(const std::vector<std::string>& args, const std::string& name, double fallback) {
    for (int i = 0; i + 1 < static_cast<int>(args.size()); ++i) {
        if (args[i] == name) {
            return std::stod(args[i + 1]);
        }
    }
    return fallback;
}

unsigned int parseUintArg(const std::vector<std::string>& args, const std::string& name, unsigned int fallback) {
    for (int i = 0; i + 1 < static_cast<int>(args.size()); ++i) {
        if (args[i] == name) {
            return static_cast<unsigned int>(std::stoul(args[i + 1]));
        }
    }
    return fallback;
}

bool hasFlag(const std::vector<std::string>& args, const std::string& name) {
    return std::find(args.begin(), args.end(), name) != args.end();
}

// Lattice and basis
std::vector<Bond> makeSquareLatticeBonds(int latticeSize, BoundaryCondition boundary) {
    std::vector<Bond> bonds;
    int expectedBondCount = (boundary == BoundaryCondition::Periodic)
        ? 2 * latticeSize * latticeSize
        : 2 * latticeSize * (latticeSize - 1);
    bonds.reserve(std::max(0, expectedBondCount));

    auto site = [latticeSize](int x, int y) {
        return y * latticeSize + x;
    };

    for (int y = 0; y < latticeSize; ++y) {
        for (int x = 0; x < latticeSize; ++x) {
            int i = site(x, y);

            if (x + 1 < latticeSize) {
                bonds.push_back({i, site(x + 1, y)});
            } else if (boundary == BoundaryCondition::Periodic) {
                bonds.push_back({i, site(0, y)});
            }

            if (y + 1 < latticeSize) {
                bonds.push_back({i, site(x, y + 1)});
            } else if (boundary == BoundaryCondition::Periodic) {
                bonds.push_back({i, site(x, 0)});
            }
        }
    }

    return bonds;
}

std::vector<Mask> makeFixedParticleBasis(int siteCount, int particleCount) {
    if (siteCount > 31) {
        throw std::runtime_error("This compact bit-mask implementation assumes siteCount <= 31.");
    }

    std::vector<Mask> basis;
    if (particleCount < 0 || particleCount > siteCount) {
        return basis;
    }

    Mask maxState = (1U << siteCount);
    for (Mask state = 0; state < maxState; ++state) {
        if (popcountMask(state) == particleCount) {
            basis.push_back(state);
        }
    }

    return basis;
}

// Local matrix elements
int countOccupiedBefore(Mask state, int site) {
    int count = 0;
    for (int k = 0; k < site; ++k) {
        if (occupied(state, k)) {
            count += 1;
        }
    }
    return count;
}

int fermionHopSign(Mask state, int fromSite, int toSite) {
    int signPower = countOccupiedBefore(state, fromSite);
    Mask stateAfterAnnihilation = state ^ (1U << fromSite);
    signPower += countOccupiedBefore(stateAfterAnnihilation, toSite);
    return (signPower % 2 == 0) ? 1 : -1;
}

LocalMatrixElement applyHardcoreBosonOperator(
    Mask state,
    const BondOperator& op,
    const std::vector<Bond>& bonds
) {
    LocalMatrixElement out;

    if (op.kind == OperatorKind::Identity) {
        out.valid = true;
        out.newState = state;
        out.absWeight = 1.0;
        return out;
    }

    int i = bonds[op.bond].first;
    int j = bonds[op.bond].second;
    bool ni = occupied(state, i);
    bool nj = occupied(state, j);

    if (op.kind == OperatorKind::Diagonal) {
        double weight = 2.0 - static_cast<double>(ni) - static_cast<double>(nj);
        if (weight <= 0.0) {
            return out;
        }
        out.valid = true;
        out.newState = state;
        out.absWeight = weight;
        return out;
    }

    if (op.kind == OperatorKind::Hopping) {
        if (ni == nj) {
            return out;
        }
        out.valid = true;
        out.newState = flipTwoSites(state, i, j);
        out.absWeight = 1.0;
        return out;
    }

    throw std::runtime_error("Unknown operator kind.");
}

LocalMatrixElement applySpinlessFermionOperator(
    Mask state,
    const BondOperator& op,
    const std::vector<Bond>& bonds
) {
    LocalMatrixElement out;

    if (op.kind == OperatorKind::Identity) {
        out.valid = true;
        out.newState = state;
        out.absWeight = 1.0;
        out.sign = 1;
        return out;
    }

    int i = bonds[op.bond].first;
    int j = bonds[op.bond].second;
    bool ni = occupied(state, i);
    bool nj = occupied(state, j);

    if (op.kind == OperatorKind::Diagonal) {
        double weight = 2.0 - static_cast<double>(ni) - static_cast<double>(nj);
        if (weight <= 0.0) {
            return out;
        }
        out.valid = true;
        out.newState = state;
        out.absWeight = weight;
        out.sign = 1;
        return out;
    }

    if (op.kind == OperatorKind::Hopping) {
        if (ni == nj) {
            return out;
        }

        int fromSite = ni ? i : j;
        int toSite = ni ? j : i;

        out.valid = true;
        out.newState = flipTwoSites(state, i, j);
        out.absWeight = 1.0;
        out.sign = fermionHopSign(state, fromSite, toSite);
        return out;
    }

    throw std::runtime_error("Unknown operator kind.");
}

LocalMatrixElement applyModelOperator(
    Mask state,
    const BondOperator& op,
    const std::vector<Bond>& bonds,
    double hoppingT,
    ParticleModel model,
    const sho::HarmonicPotential& trap
) {
    if (op.kind == OperatorKind::Potential) {
        LocalMatrixElement out;
        const double shiftedWeight = trap.shiftedMatrixElement(op.bond, occupied(state, op.bond));

        if (!trap.enabled || shiftedWeight <= 0.0) {
            return out;
        }

        out.valid = true;
        out.newState = state;

        // The global SSE factor remains (beta * t)^n, so divide the
        // potential's energy-valued matrix element by t here.
        out.absWeight = shiftedWeight / hoppingT;
        out.sign = 1;
        return out;
    }

    if (model == ParticleModel::HardcoreBoson) {
        return applyHardcoreBosonOperator(state, op, bonds);
    }
    return applySpinlessFermionOperator(state, op, bonds);
}

// Configuration weight
SignedLogWeight signedLogConfigurationWeight(
    Mask initialState,
    const std::vector<BondOperator>& operators,
    const std::vector<Bond>& bonds,
    double beta,
    double hoppingT,
    ParticleModel model,
    const sho::HarmonicPotential& trap
) {
    int stringLength = static_cast<int>(operators.size());
    int activeCount = 0;
    double logLocalProduct = 0.0;
    int totalSign = 1;

    Mask state = initialState;

    for (const BondOperator& op : operators) {
        if (op.kind == OperatorKind::Identity) {
            continue;
        }

        activeCount += 1;
        LocalMatrixElement elem = applyModelOperator(
            state,
            op,
            bonds,
            hoppingT,
            model,
            trap
        );

        if (!elem.valid || elem.absWeight <= 0.0) {
            return SignedLogWeight{false, negativeInfinity, 1, activeCount};
        }

        logLocalProduct += std::log(elem.absWeight);
        totalSign *= elem.sign;
        state = elem.newState;
    }

    if (beta * hoppingT <= 0.0) {
        throw std::runtime_error("beta and hoppingT must be positive.");
    }

    double logAbsWeight =
        static_cast<double>(activeCount) * std::log(beta * hoppingT)
        + std::lgamma(static_cast<double>(stringLength - activeCount + 1))
        - std::lgamma(static_cast<double>(stringLength + 1))
        + logLocalProduct;

    return SignedLogWeight{true, logAbsWeight, totalSign, activeCount};
}

double imaginaryTimeAveragePotential(
    Mask initialState,
    const std::vector<BondOperator>& operators,
    const std::vector<Bond>& bonds,
    const sho::HarmonicPotential& trap
) {
    if (!trap.enabled || operators.empty()) {
        return 0.0;
    }

    Mask state = initialState;
    double currentPotential = trap.statePotentialEnergy(state);
    double potentialTotal = 0.0;

    for (const BondOperator& op : operators) {
        potentialTotal += currentPotential;

        if (op.kind == OperatorKind::Hopping) {
            const int i = bonds[op.bond].first;
            const int j = bonds[op.bond].second;
            const bool particleMovesFromI = occupied(state, i);

            currentPotential += particleMovesFromI
                ? trap.valueAt(j) - trap.valueAt(i)
                : trap.valueAt(i) - trap.valueAt(j);
            state = flipTwoSites(state, i, j);
        }
    }

    return potentialTotal / static_cast<double>(operators.size());
}

bool acceptFromLogRatio(double logRatio, std::mt19937& rng) {
    if (std::isnan(logRatio)) {
        return false;
    }
    if (logRatio >= 0.0) {
        return true;
    }

    std::uniform_real_distribution<double> uniform01(0.0, 1.0);
    return std::log(uniform01(rng)) < logRatio;
}

// Updates
bool diagonalUpdateAttempt(
    Mask initialState,
    std::vector<BondOperator>& operators,
    const std::vector<Bond>& bonds,
    double beta,
    double hoppingT,
    ParticleModel model,
    const sho::HarmonicPotential& trap,
    SignedLogWeight& currentWeight,
    AcceptanceCounter& counter,
    std::mt19937& rng
) {
    counter.attempts += 1;

    int stringLength = static_cast<int>(operators.size());
    int bondCount = static_cast<int>(bonds.size());
    int siteCount = trap.enabled ? static_cast<int>(trap.sitePotential.size()) : 0;
    int diagonalChoiceCount = bondCount + siteCount;

    std::uniform_int_distribution<int> positionDist(0, stringLength - 1);
    std::uniform_int_distribution<int> diagonalChoiceDist(0, diagonalChoiceCount - 1);

    int position = positionDist(rng);
    BondOperator oldOperator = operators[position];
    BondOperator newOperator = oldOperator;
    double proposalLogRatio = 0.0;

    if (oldOperator.kind == OperatorKind::Identity) {
        const int choice = diagonalChoiceDist(rng);
        if (choice < bondCount) {
            newOperator.kind = OperatorKind::Diagonal;
            newOperator.bond = choice;
        } else {
            newOperator.kind = OperatorKind::Potential;
            newOperator.bond = choice - bondCount;
        }
        proposalLogRatio = std::log(static_cast<double>(diagonalChoiceCount));
    } else if (oldOperator.kind == OperatorKind::Diagonal ||
               oldOperator.kind == OperatorKind::Potential) {
        newOperator.kind = OperatorKind::Identity;
        newOperator.bond = -1;
        proposalLogRatio = -std::log(static_cast<double>(diagonalChoiceCount));
    } else {
        return false;
    }

    operators[position] = newOperator;

    SignedLogWeight newWeight = signedLogConfigurationWeight(
        initialState,
        operators,
        bonds,
        beta,
        hoppingT,
        model,
        trap
    );

    double logRatio = newWeight.logAbsWeight - currentWeight.logAbsWeight + proposalLogRatio;

    if (newWeight.valid && acceptFromLogRatio(logRatio, rng)) {
        currentWeight = newWeight;
        counter.accepted += 1;
        return true;
    }

    operators[position] = oldOperator;
    return false;
}

bool wormVertexUpdateAttempt(
    Mask initialState,
    std::vector<BondOperator>& operators,
    const std::vector<Bond>& bonds,
    double beta,
    double hoppingT,
    ParticleModel model,
    const sho::HarmonicPotential& trap,
    SignedLogWeight& currentWeight,
    AcceptanceCounter& counter,
    std::mt19937& rng
) {
    counter.attempts += 1;

    int stringLength = static_cast<int>(operators.size());
    std::uniform_int_distribution<int> positionDist(0, stringLength - 1);
    int position = positionDist(rng);

    BondOperator oldOperator = operators[position];
    if (oldOperator.kind == OperatorKind::Identity) {
        return false;
    }

    BondOperator newOperator = oldOperator;
    if (oldOperator.kind == OperatorKind::Diagonal) {
        newOperator.kind = OperatorKind::Hopping;
    } else if (oldOperator.kind == OperatorKind::Hopping) {
        newOperator.kind = OperatorKind::Diagonal;
    } else {
        return false;
    }

    operators[position] = newOperator;

    SignedLogWeight newWeight = signedLogConfigurationWeight(
        initialState,
        operators,
        bonds,
        beta,
        hoppingT,
        model,
        trap
    );

    double logRatio = newWeight.logAbsWeight - currentWeight.logAbsWeight;

    if (newWeight.valid && acceptFromLogRatio(logRatio, rng)) {
        currentWeight = newWeight;
        counter.accepted += 1;
        return true;
    }

    operators[position] = oldOperator;
    return false;
}

bool multiVertexWormUpdateAttempt(
    Mask initialState,
    std::vector<BondOperator>& operators,
    const std::vector<Bond>& bonds,
    double beta,
    double hoppingT,
    ParticleModel model,
    const sho::HarmonicPotential& trap,
    SignedLogWeight& currentWeight,
    AcceptanceCounter& counter,
    std::mt19937& rng
) {
    counter.attempts += 1;

    int stringLength = static_cast<int>(operators.size());
    std::uniform_int_distribution<int> lengthDist(1, 6);
    std::uniform_int_distribution<int> positionDist(0, stringLength - 1);

    int wormLength = lengthDist(rng);
    std::array<int, 6> positions{};
    int positionCount = 0;

    for (int attempt = 0; attempt < 10 * wormLength && positionCount < wormLength; ++attempt) {
        int position = positionDist(rng);
        if (operators[position].kind == OperatorKind::Diagonal ||
            operators[position].kind == OperatorKind::Hopping) {
            positions[positionCount] = position;
            positionCount++;
        }
    }

    if (positionCount == 0) {
        return false;
    }

    std::array<BondOperator, 6> oldValues{};

    for (int idx = 0; idx < positionCount; ++idx) {
        int pos = positions[idx];
        oldValues[idx] = operators[pos];

        if (operators[pos].kind == OperatorKind::Diagonal) {
            operators[pos].kind = OperatorKind::Hopping;
        } else if (operators[pos].kind == OperatorKind::Hopping) {
            operators[pos].kind = OperatorKind::Diagonal;
        }
    }

    SignedLogWeight newWeight = signedLogConfigurationWeight(
        initialState,
        operators,
        bonds,
        beta,
        hoppingT,
        model,
        trap
    );

    double logRatio = newWeight.logAbsWeight - currentWeight.logAbsWeight;

    if (newWeight.valid && acceptFromLogRatio(logRatio, rng)) {
        currentWeight = newWeight;
        counter.accepted += 1;
        return true;
    }

    for (int idx = 0; idx < positionCount; ++idx) {
        operators[positions[idx]] = oldValues[idx];
    }
    return false;
}

bool boundaryLocalHopUpdateAttempt(
    Mask& initialState,
    std::vector<BondOperator>& operators,
    const std::vector<Bond>& bonds,
    double beta,
    double hoppingT,
    ParticleModel model,
    const sho::HarmonicPotential& trap,
    SignedLogWeight& currentWeight,
    AcceptanceCounter& counter,
    std::mt19937& rng
) {
    counter.attempts += 1;

    std::uniform_int_distribution<int> bondDist(0, static_cast<int>(bonds.size()) - 1);
    int bondIndex = bondDist(rng);
    int i = bonds[bondIndex].first;
    int j = bonds[bondIndex].second;

    bool ni = occupied(initialState, i);
    bool nj = occupied(initialState, j);
    if (ni == nj) {
        return false;
    }

    Mask oldState = initialState;
    Mask newState = flipTwoSites(initialState, i, j);

    SignedLogWeight newWeight = signedLogConfigurationWeight(
        newState,
        operators,
        bonds,
        beta,
        hoppingT,
        model,
        trap
    );

    double logRatio = newWeight.logAbsWeight - currentWeight.logAbsWeight;

    if (newWeight.valid && acceptFromLogRatio(logRatio, rng)) {
        initialState = newState;
        currentWeight = newWeight;
        counter.accepted += 1;
        return true;
    }

    initialState = oldState;
    return false;
}

bool boundaryStateUpdateAttempt(
    Mask& initialState,
    std::vector<BondOperator>& operators,
    const std::vector<Mask>& basis,
    const std::vector<Bond>& bonds,
    double beta,
    double hoppingT,
    ParticleModel model,
    const sho::HarmonicPotential& trap,
    SignedLogWeight& currentWeight,
    AcceptanceCounter& counter,
    std::mt19937& rng
) {
    counter.attempts += 1;

    std::uniform_int_distribution<int> basisDist(0, static_cast<int>(basis.size()) - 1);
    Mask oldState = initialState;
    Mask newState = basis[basisDist(rng)];

    if (newState == oldState) {
        return false;
    }

    SignedLogWeight newWeight = signedLogConfigurationWeight(
        newState,
        operators,
        bonds,
        beta,
        hoppingT,
        model,
        trap
    );

    double logRatio = newWeight.logAbsWeight - currentWeight.logAbsWeight;

    if (newWeight.valid && acceptFromLogRatio(logRatio, rng)) {
        initialState = newState;
        currentWeight = newWeight;
        counter.accepted += 1;
        return true;
    }

    return false;
}

// Statistics and progress
double meanValue(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    double total = std::accumulate(values.begin(), values.end(), 0.0);
    return total / static_cast<double>(values.size());
}

double standardError(const std::vector<double>& values) {
    if (values.size() < 2) {
        return 0.0;
    }

    double avg = meanValue(values);
    double variance = 0.0;
    for (double x : values) {
        double dx = x - avg;
        variance += dx * dx;
    }
    variance /= static_cast<double>(values.size() - 1);
    return std::sqrt(variance / static_cast<double>(values.size()));
}

std::vector<MeasurementBlock> makeMeasurementBlocks(long long measurementSweeps, int measureEvery, int requestedBlocks) {
    long long measurementCount = (measurementSweeps + measureEvery - 1) / measureEvery;
    int actualBlocks = static_cast<int>(std::max<long long>(1, std::min<long long>(requestedBlocks, measurementCount)));
    return std::vector<MeasurementBlock>(actualBlocks);
}

int measurementBlockIndex(long long measurementIndex, long long totalMeasurements, int blockCount) {
    if (blockCount <= 1 || totalMeasurements <= 1) {
        return 0;
    }
    long long blockIndex = measurementIndex * blockCount / totalMeasurements;
    return static_cast<int>(std::min<long long>(blockIndex, blockCount - 1));
}

void printProgressBar(
    long long current,
    long long total,
    int particleCount,
    const std::chrono::steady_clock::time_point& startTime,
    const std::string& unitName = "sweeps"
) {
    (void)startTime;

    const int barWidth = 40;
    double progress = total == 0 ? 1.0 : static_cast<double>(current) / static_cast<double>(total);
    progress = std::max(0.0, std::min(1.0, progress));

    int filled = static_cast<int>(progress * barWidth);
    int percent = static_cast<int>(progress * 100.0);

    std::cerr << "\rRunning N=" << particleCount << " [";

    for (int i = 0; i < barWidth; ++i) {
        if (i < filled) {
            std::cerr << "=";
        } else if (i == filled && current < total) {
            std::cerr << ">";
        } else {
            std::cerr << " ";
        }
    }

    std::cerr << "] "
              << std::setw(3) << percent << "% "
              << "(" << current << "/" << total << " " << unitName << ")"
              << std::flush;
}

void printParallelChainProgress(
    std::atomic<long long>* chainProgress,
    int chainCount,
    long long totalSweeps,
    int particleCount,
    bool firstDraw = false
) {
    const int barWidth = 30;
    const int linesToRedraw = chainCount + 1;

    if (!firstDraw) {
        std::cerr << "\033[" << linesToRedraw << "A";
    }

    std::cerr << "\033[KRunning N=" << particleCount
              << " | per-chain progress\n";

    for (int chain = 0; chain < chainCount; ++chain) {
        long long current = chainProgress[chain].load();
        double progress = totalSweeps == 0
            ? 1.0
            : static_cast<double>(current) / static_cast<double>(totalSweeps);
        progress = std::max(0.0, std::min(1.0, progress));

        int filled = static_cast<int>(progress * barWidth);
        int percent = static_cast<int>(progress * 100.0);

        std::cerr << "\033[Kchain " << std::setw(2) << chain << " [";
        for (int i = 0; i < barWidth; ++i) {
            if (i < filled) {
                std::cerr << "=";
            } else if (i == filled && current < totalSweeps) {
                std::cerr << ">";
            } else {
                std::cerr << " ";
            }
        }
        std::cerr << "] " << std::setw(3) << percent << "% "
                  << "(" << current << "/" << totalSweeps << ")\n";
    }

    std::cerr << std::flush;
}

// Main QMC routines
QmcResult runWormQmc(const QmcParameters& params);
QmcResult combineChainResults(const std::vector<QmcResult>& results);

QmcResult runParallelChains(const QmcParameters& baseParams, int chains) {
    if (chains <= 0) {
        throw std::runtime_error("--chains must be positive.");
    }

    std::vector<QmcResult> results(chains);
    std::vector<std::atomic<long long>> chainProgress(chains);
    for (int chain = 0; chain < chains; ++chain) {
        chainProgress[chain].store(0);
    }

    long long totalSweeps = baseParams.thermalizationSweeps + baseParams.measurementSweeps;
    printParallelChainProgress(chainProgress.data(), chains, totalSweeps, baseParams.particleCount, true);

    #pragma omp parallel for schedule(dynamic)
    for (int chain = 0; chain < chains; ++chain) {
        QmcParameters localParams = baseParams;
        localParams.seed = baseParams.seed + static_cast<unsigned int>(1000003 * (chain + 1));
        localParams.showProgress = false;
        localParams.sharedChainProgress = chainProgress.data();
        localParams.chainIndex = chain;
        localParams.chainCount = chains;

        results[chain] = runWormQmc(localParams);
    }

    printParallelChainProgress(chainProgress.data(), chains, totalSweeps, baseParams.particleCount);
    std::cerr << "\nPer-chain energies:\n";
    for (int chain = 0; chain < chains; ++chain) {
        std::cerr << "chain " << chain
                  << ": E = " << std::fixed << std::setprecision(8)
                  << results[chain].energy
                  << ", err = " << results[chain].energyError
                  << ", <Nop> = " << results[chain].averageActive
                  << ", <P> = " << results[chain].averagePotentialOperators
                  << ", <V> = " << results[chain].averagePotentialEnergy
                  << ", <sgn> = " << results[chain].averageSign
                  << " +/- " << results[chain].signError
                  << ", accD = " << results[chain].diagonalAcceptance
                  << ", accW = " << results[chain].wormAcceptance
                  << ", accS = " << results[chain].stateAcceptance
                  << "\n";
    }

    return combineChainResults(results);
}

QmcResult runWormQmc(const QmcParameters& params) {
    if (params.latticeSize <= 0) {
        throw std::runtime_error("latticeSize must be positive.");
    }
    if (params.stringLength <= 0) {
        throw std::runtime_error("stringLength must be positive.");
    }
    if (params.measureEvery <= 0) {
        throw std::runtime_error("measureEvery must be positive.");
    }

    int siteCount = params.latticeSize * params.latticeSize;
    sho::HarmonicPotential trap = sho::makeHarmonicPotential(
        params.latticeSize,
        params.harmonicTrap
    );
    std::vector<Bond> bonds = makeSquareLatticeBonds(params.latticeSize, params.boundary);
    std::vector<Mask> basis = makeFixedParticleBasis(siteCount, params.particleCount);

    if (basis.empty()) {
        throw std::runtime_error("Empty fixed-particle-number basis.");
    }

    std::mt19937 rng(params.seed);
    std::uniform_int_distribution<int> basisDist(0, static_cast<int>(basis.size()) - 1);

    Mask initialState = basis[basisDist(rng)];
    std::vector<BondOperator> operators(params.stringLength, BondOperator{OperatorKind::Identity, -1});

    SignedLogWeight currentWeight = signedLogConfigurationWeight(
        initialState,
        operators,
        bonds,
        params.beta,
        params.hoppingT,
        params.model,
        trap
    );

    if (!currentWeight.valid) {
        throw std::runtime_error("Initial configuration has invalid weight.");
    }

    AcceptanceCounter diagonalCounter;
    AcceptanceCounter wormCounter;
    AcceptanceCounter stateCounter;

    long long totalSweeps = params.thermalizationSweeps + params.measurementSweeps;
    long long progressInterval = std::max<long long>(1, totalSweeps / 100);
    long long totalMeasurements = (params.measurementSweeps + params.measureEvery - 1) / params.measureEvery;
    std::vector<MeasurementBlock> blocks = makeMeasurementBlocks(
        params.measurementSweeps,
        params.measureEvery,
        params.blockCount
    );

    long long measurementIndex = 0;
    long long cutoffHits = 0;

    double activeTotal = 0.0;
    double diagonalTotal = 0.0;
    double hoppingTotal = 0.0;
    double potentialOperatorTotal = 0.0;
    double signTotal = 0.0;
    double signedActiveTotal = 0.0;
    double signedPotentialEnergyTotal = 0.0;

    auto startTime = std::chrono::steady_clock::now();

    for (long long sweep = 0; sweep < totalSweeps; ++sweep) {
        if (params.showProgress && (sweep == 0 || sweep % progressInterval == 0)) {
            printProgressBar(sweep, totalSweeps, params.particleCount, startTime, "sweeps");
        }

        if (params.sharedChainProgress != nullptr &&
            params.chainIndex >= 0 &&
            (sweep == 0 || sweep % progressInterval == 0)) {
            params.sharedChainProgress[params.chainIndex].store(sweep);
            #pragma omp critical(qmcProgressPrint)
            {
                printParallelChainProgress(
                    params.sharedChainProgress,
                    params.chainCount,
                    totalSweeps,
                    params.particleCount
                );
            }
        }

        for (int i = 0; i < params.stringLength; ++i) {
            diagonalUpdateAttempt(
                initialState,
                operators,
                bonds,
                params.beta,
                params.hoppingT,
                params.model,
                trap,
                currentWeight,
                diagonalCounter,
                rng
            );
        }

        for (int i = 0; i < params.stringLength; ++i) {
            wormVertexUpdateAttempt(
                initialState,
                operators,
                bonds,
                params.beta,
                params.hoppingT,
                params.model,
                trap,
                currentWeight,
                wormCounter,
                rng
            );
        }

        for (int i = 0; i < siteCount; ++i) {
            boundaryLocalHopUpdateAttempt(
                initialState,
                operators,
                bonds,
                params.beta,
                params.hoppingT,
                params.model,
                trap,
                currentWeight,
                stateCounter,
                rng
            );
        }

        for (int i = 0; i < 2; ++i) {
            boundaryStateUpdateAttempt(
                initialState,
                operators,
                basis,
                bonds,
                params.beta,
                params.hoppingT,
                params.model,
                trap,
                currentWeight,
                stateCounter,
                rng
            );
        }

        if (sweep >= params.thermalizationSweeps &&
            ((sweep - params.thermalizationSweeps) % params.measureEvery == 0)) {
            int active = currentWeight.activeCount;
            int diagonal = 0;
            int hopping = 0;
            int potentialOperators = 0;

            for (const BondOperator& op : operators) {
                if (op.kind == OperatorKind::Diagonal) {
                    diagonal += 1;
                } else if (op.kind == OperatorKind::Hopping) {
                    hopping += 1;
                } else if (op.kind == OperatorKind::Potential) {
                    potentialOperators += 1;
                }
            }

            double sign = static_cast<double>(currentWeight.sign);
            double signedActive = sign * static_cast<double>(active);
            double potentialEnergy = imaginaryTimeAveragePotential(
                initialState,
                operators,
                bonds,
                trap
            );
            double signedPotentialEnergy = sign * potentialEnergy;

            activeTotal += active;
            diagonalTotal += diagonal;
            hoppingTotal += hopping;
            potentialOperatorTotal += potentialOperators;
            signTotal += sign;
            signedActiveTotal += signedActive;
            signedPotentialEnergyTotal += signedPotentialEnergy;

            int blockIndex = measurementBlockIndex(measurementIndex, totalMeasurements, static_cast<int>(blocks.size()));
            blocks[blockIndex].count += 1;
            blocks[blockIndex].activeSum += active;
            blocks[blockIndex].diagonalSum += diagonal;
            blocks[blockIndex].hoppingSum += hopping;
            blocks[blockIndex].potentialOperatorSum += potentialOperators;
            blocks[blockIndex].signSum += sign;
            blocks[blockIndex].signedActiveSum += signedActive;
            blocks[blockIndex].signedPotentialEnergySum += signedPotentialEnergy;

            if (active > static_cast<int>(0.85 * params.stringLength)) {
                cutoffHits += 1;
            }

            measurementIndex += 1;
        }
    }

    if (params.sharedChainProgress != nullptr && params.chainIndex >= 0) {
        params.sharedChainProgress[params.chainIndex].store(totalSweeps);
        #pragma omp critical(qmcProgressPrint)
        {
            printParallelChainProgress(
                params.sharedChainProgress,
                params.chainCount,
                totalSweeps,
                params.particleCount
            );
        }
    }

    if (params.showProgress) {
        printProgressBar(totalSweeps, totalSweeps, params.particleCount, startTime, "sweeps");
        std::cerr << "\n";
    }

    if (measurementIndex == 0) {
        throw std::runtime_error("No measurements were recorded.");
    }

    std::vector<double> blockEnergies;
    std::vector<double> blockSigns;
    for (const MeasurementBlock& block : blocks) {
        if (block.count == 0) {
            continue;
        }

        double blockAverageSign = block.signSum / static_cast<double>(block.count);
        blockSigns.push_back(blockAverageSign);

        if (std::abs(block.signSum) < 1e-14) {
            continue;
        }
        blockEnergies.push_back(
            -block.signedActiveSum / (params.beta * block.signSum)
            + trap.totalShift
        );
    }

    QmcResult result;
    result.latticeSize = params.latticeSize;
    result.particleCount = params.particleCount;
    result.beta = params.beta;
    result.hoppingT = params.hoppingT;
    result.stringLength = params.stringLength;
    result.basisSize = static_cast<int>(basis.size());
    result.averageActive = activeTotal / static_cast<double>(measurementIndex);
    result.averageDiagonal = diagonalTotal / static_cast<double>(measurementIndex);
    result.averageHopping = hoppingTotal / static_cast<double>(measurementIndex);
    result.averagePotentialOperators =
        potentialOperatorTotal / static_cast<double>(measurementIndex);
    result.averageSign = signTotal / static_cast<double>(measurementIndex);
    result.signError = standardError(blockSigns);

    if (std::abs(signTotal) < 1e-14) {
        result.energy = std::numeric_limits<double>::quiet_NaN();
        result.energyError = std::numeric_limits<double>::quiet_NaN();
        result.averagePotentialEnergy = std::numeric_limits<double>::quiet_NaN();
    } else {
        result.energy =
            -signedActiveTotal / (params.beta * signTotal)
            + trap.totalShift;
        result.energyError = standardError(blockEnergies);
        result.averagePotentialEnergy = signedPotentialEnergyTotal / signTotal;
    }

    result.cutoffHitFraction = static_cast<double>(cutoffHits) / static_cast<double>(measurementIndex);
    result.diagonalAcceptance = diagonalCounter.attempts == 0
        ? 0.0
        : static_cast<double>(diagonalCounter.accepted) / static_cast<double>(diagonalCounter.attempts);
    result.wormAcceptance = wormCounter.attempts == 0
        ? 0.0
        : static_cast<double>(wormCounter.accepted) / static_cast<double>(wormCounter.attempts);
    result.stateAcceptance = stateCounter.attempts == 0
        ? 0.0
        : static_cast<double>(stateCounter.accepted) / static_cast<double>(stateCounter.attempts);

    return result;
}

QmcResult combineChainResults(const std::vector<QmcResult>& results) {
    if (results.empty()) {
        throw std::runtime_error("No chain results to combine.");
    }

    QmcResult combined = results[0];
    int chains = static_cast<int>(results.size());

    combined.averageActive = 0.0;
    combined.averageDiagonal = 0.0;
    combined.averageHopping = 0.0;
    combined.averagePotentialOperators = 0.0;
    combined.averagePotentialEnergy = 0.0;
    combined.averageSign = 0.0;
    combined.signError = 0.0;
    combined.energy = 0.0;
    combined.diagonalAcceptance = 0.0;
    combined.wormAcceptance = 0.0;
    combined.stateAcceptance = 0.0;
    combined.cutoffHitFraction = 0.0;

    std::vector<double> chainEnergies;
    std::vector<double> chainSigns;
    chainEnergies.reserve(chains);
    chainSigns.reserve(chains);

    for (const QmcResult& result : results) {
        combined.averageActive += result.averageActive;
        combined.averageDiagonal += result.averageDiagonal;
        combined.averageHopping += result.averageHopping;
        combined.averagePotentialOperators += result.averagePotentialOperators;
        combined.averagePotentialEnergy += result.averagePotentialEnergy;
        combined.averageSign += result.averageSign;
        combined.energy += result.energy;
        combined.diagonalAcceptance += result.diagonalAcceptance;
        combined.wormAcceptance += result.wormAcceptance;
        combined.stateAcceptance += result.stateAcceptance;
        combined.cutoffHitFraction += result.cutoffHitFraction;
        chainEnergies.push_back(result.energy);
        chainSigns.push_back(result.averageSign);
    }

    combined.averageActive /= chains;
    combined.averageDiagonal /= chains;
    combined.averageHopping /= chains;
    combined.averagePotentialOperators /= chains;
    combined.averagePotentialEnergy /= chains;
    combined.averageSign /= chains;
    combined.energy /= chains;
    combined.diagonalAcceptance /= chains;
    combined.wormAcceptance /= chains;
    combined.stateAcceptance /= chains;
    combined.cutoffHitFraction /= chains;

    if (chains > 1) {
        combined.energyError = standardError(chainEnergies);
        combined.signError = standardError(chainSigns);
    } else {
        combined.energyError = results[0].energyError;
        combined.signError = results[0].signError;
    }

    return combined;
}

// Output
double referenceEnergyValue(
    ParticleModel model,
    BoundaryCondition boundary,
    int latticeSize,
    double beta,
    double hoppingT,
    int particleCount
) {
    if (boundary != BoundaryCondition::Periodic ||
        latticeSize != 4 ||
        std::abs(beta - 1.0) >= 1e-12 ||
        std::abs(hoppingT - 1.0) >= 1e-12) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (model == ParticleModel::HardcoreBoson) {
        const std::vector<double> bosonValues = {
            -63.56914,
            -62.66398,
            -61.25074,
            -59.29942,
            -56.78704,
            -53.69263,
            -49.99806
        };

        if (particleCount >= 2 && particleCount <= 8) {
            return bosonValues[particleCount - 2];
        }
    }

    if (model == ParticleModel::SpinlessFermion) {
        switch (particleCount) {
            case 2: return -61.864;
            case 3: return -59.861;
            case 4: return -57.107;
            case 5: return -55.593;
            case 8: return -43.488;
            default: break;
        }
    }

    return std::numeric_limits<double>::quiet_NaN();
}

void printResultRow(const QmcResult& result, double exactValue) {
    std::cout << std::left << std::setw(4) << result.particleCount
              << std::right << std::fixed << std::setprecision(6)
              << std::setw(14) << result.energy
              << " +/- " << std::setw(9) << result.energyError
              << std::setw(14) << exactValue
              << std::setw(14) << (result.energy - exactValue)
              << std::setw(10) << result.averageActive
              << std::setw(10) << result.averageDiagonal
              << std::setw(10) << result.averageHopping
              << std::setw(10) << result.averagePotentialOperators
              << std::setw(12) << result.averagePotentialEnergy
              << std::setw(10) << result.averageSign
              << std::setw(10) << result.signError
              << std::setw(10) << result.diagonalAcceptance
              << std::setw(10) << result.wormAcceptance
              << std::setw(10) << result.stateAcceptance
              << "\n";
}

void printHelp() {
    std::cout << "Hard-core boson / spinless fermion stochastic worm QMC sampler\n\n";
    std::cout << "Usage examples:\n";
    std::cout << "  ./worm_qmc --model boson --N 2 --chains 12\n";
    std::cout << "  ./worm_qmc --model fermion --boundary open --N 2 --chains 12\n";
    std::cout << "  ./worm_qmc --all --therm 10000 --sweeps 100000 --M 220\n\n";
    std::cout << "Options:\n";
    std::cout << "  --model <boson|fermion>       particle model, default boson\n";
    std::cout << "  --boundary <periodic|open>    lattice boundary, default periodic\n";
    std::cout << "  --all                         run N = 2 through 8\n";
    std::cout << "  --N <int>                     particle number for a single run\n";
    std::cout << "  --L <int>                     square lattice side length, default 4\n";
    std::cout << "  --beta <float>                inverse temperature/projection length\n";
    std::cout << "  --t <float>                   hopping t, default 1\n";
    std::cout << "  --trap                        enable the harmonic trap\n";
    std::cout << "  --mass <float>                particle mass m, default 1\n";
    std::cout << "  --omega <float>               trap angular frequency, default 1\n";
    std::cout << "  --a <float>                   lattice spacing, default 1\n";
    std::cout << "  --center_x <float>            trap center x, default (L-1)/2\n";
    std::cout << "  --center_y <float>            trap center y, default (L-1)/2\n";
    std::cout << "  --trap_shift <float>          positive SSE shift epsilon, default 1e-12\n";
    std::cout << "  --M <int>                     SSE operator-string cutoff\n";
    std::cout << "  --therm <int>                 thermalization sweeps\n";
    std::cout << "  --sweeps <int>                measurement sweeps\n";
    std::cout << "  --chains <int>                independent chains\n";
    std::cout << "  --seed <int>                  random seed\n";
    std::cout << "  --random_seed                 ignore --seed and use std::random_device\n";
}

// Main
int main(int argc, char** argv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        args.push_back(std::string(argv[i]));
    }

    if (hasFlag(args, "--help")) {
        printHelp();
        return 0;
    }

    QmcParameters params;
    params.model = parseParticleModel(args, params.model);
    params.boundary = parseBoundaryCondition(args, params.boundary);
    params.latticeSize = parseIntArg(args, "--L", params.latticeSize);
    params.particleCount = parseIntArg(args, "--N", params.particleCount);
    params.beta = parseDoubleArg(args, "--beta", params.beta);
    params.hoppingT = parseDoubleArg(args, "--t", params.hoppingT);
    params.harmonicTrap.enabled = hasFlag(args, "--trap");
    params.harmonicTrap.mass =
        parseDoubleArg(args, "--mass", params.harmonicTrap.mass);
    params.harmonicTrap.omega =
        parseDoubleArg(args, "--omega", params.harmonicTrap.omega);
    params.harmonicTrap.latticeSpacing =
        parseDoubleArg(args, "--a", params.harmonicTrap.latticeSpacing);
    params.harmonicTrap.centerX =
        parseDoubleArg(args, "--center_x", params.harmonicTrap.centerX);
    params.harmonicTrap.centerY =
        parseDoubleArg(args, "--center_y", params.harmonicTrap.centerY);
    params.harmonicTrap.shiftEpsilon =
        parseDoubleArg(args, "--trap_shift", params.harmonicTrap.shiftEpsilon);
    params.stringLength = parseIntArg(args, "--M", params.stringLength);
    params.thermalizationSweeps = parseLongLongArg(args, "--therm", params.thermalizationSweeps);
    params.measurementSweeps = parseLongLongArg(args, "--sweeps", params.measurementSweeps);
    params.measureEvery = parseIntArg(args, "--measure_every", params.measureEvery);
    params.blockCount = parseIntArg(args, "--blocks", params.blockCount);
    params.seed = parseUintArg(args, "--seed", params.seed);

    if (hasFlag(args, "--random_seed")) {
        std::random_device rd;
        params.seed = rd();
    }

    int chains = parseIntArg(args, "--chains", 1);

    std::vector<int> particleNumbers;
    if (hasFlag(args, "--all")) {
        particleNumbers = {2, 3, 4, 5, 6, 7, 8};
    } else {
        particleNumbers = {params.particleCount};
    }


    std::cout << "Stochastic worm QMC sampler\n";
    std::cout << "model=" << modelName(params.model)
              << ", boundary=" << boundaryName(params.boundary)
              << ", L=" << params.latticeSize
              << ", beta=" << params.beta
              << ", t=" << params.hoppingT
              << ", trap=" << (params.harmonicTrap.enabled ? "on" : "off")
              << ", M=" << params.stringLength
              << ", therm=" << params.thermalizationSweeps
              << ", sweeps=" << params.measurementSweeps
              << ", chains=" << chains
              << ", seed=" << params.seed
              << "\n";

    if (params.harmonicTrap.enabled) {
        const double defaultCenter = 0.5 * static_cast<double>(params.latticeSize - 1);
        const double centerX = std::isnan(params.harmonicTrap.centerX)
            ? defaultCenter
            : params.harmonicTrap.centerX;
        const double centerY = std::isnan(params.harmonicTrap.centerY)
            ? defaultCenter
            : params.harmonicTrap.centerY;

        std::cout << "SHO: m=" << params.harmonicTrap.mass
                  << ", omega=" << params.harmonicTrap.omega
                  << ", a=" << params.harmonicTrap.latticeSpacing
                  << ", center=(" << centerX << ", " << centerY << ")"
                  << ", shift_epsilon=" << params.harmonicTrap.shiftEpsilon
                  << "\n";

        if (params.boundary == BoundaryCondition::Periodic) {
            std::cerr << "Warning: a centered harmonic trap is usually paired with "
                      << "open boundaries; periodic hopping creates a wraparound seam.\n";
        }
    }
    std::cout << "\n";

    std::cout << std::left << std::setw(4) << "N"
              << std::right << std::setw(14) << "QMC E"
              << " +/- " << std::setw(9) << "err"
              << std::setw(14) << "Exact"
              << std::setw(14) << "diff"
              << std::setw(10) << "<Nop>"
              << std::setw(10) << "<D>"
              << std::setw(10) << "<H>"
              << std::setw(10) << "<P>"
              << std::setw(12) << "<V>"
              << std::setw(10) << "<sgn>"
              << std::setw(10) << "sgn_err"
              << std::setw(10) << "accD"
              << std::setw(10) << "accW"
              << std::setw(10) << "accS"
              << "\n";

    std::cout << std::string(171, '-') << "\n";

    for (int particleCount : particleNumbers) {
        params.particleCount = particleCount;
        params.seed += static_cast<unsigned int>(17 * particleCount);

        QmcResult result;
        if (chains == 1) {
            params.showProgress = true;
            result = runWormQmc(params);
        } else {
            result = runParallelChains(params, chains);
        }

        double exactValue = params.harmonicTrap.enabled
            ? std::numeric_limits<double>::quiet_NaN()
            : referenceEnergyValue(
                params.model,
                params.boundary,
                params.latticeSize,
                params.beta,
                params.hoppingT,
                particleCount
            );

        printResultRow(result, exactValue);

        if (result.cutoffHitFraction > 0.01) {
            std::cerr << "Warning: cutoff hit fraction for N=" << particleCount
                      << " is " << result.cutoffHitFraction
                      << ". Increase --M.\n";
        }

        if (std::abs(result.averageSign) < 0.05) {
            std::cerr << "Warning: average sign is small for N=" << particleCount
                      << " (<sgn>=" << result.averageSign << "). Fermion result may be unreliable.\n";
        }
    }

    return 0;
}
