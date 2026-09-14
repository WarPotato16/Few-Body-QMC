#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Mask = std::uint64_t;
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
    int bond = -1;
    int site = -1;
};

struct QmcParameters {
    int latticeSize = 4;
    int particleCount = 2;
    double beta = 1.0;
    double hoppingT = 1.0;
    int stringLength = 800;
    long long thermalizationSweeps = 1000;
    long long measurementSweeps = 10000;
    int measureEvery = 1;
    int blockCount = 50;
    ParticleModel model = ParticleModel::HardcoreBoson;
    BoundaryCondition boundary = BoundaryCondition::Periodic;
    unsigned int seed = 12345;
    bool showProgress = true;

    // Harmonic trapping potential V_i = 1/2 m omega^2 a^2 r_i^2.
    bool harmonicTrap = false;
    double mass = 1.0;
    double omega = 0.5;
    double latticeSpacing = 1.0;
    double trapCenterX = std::numeric_limits<double>::quiet_NaN();
    double trapCenterY = std::numeric_limits<double>::quiet_NaN();
    double trapCenterZ = std::numeric_limits<double>::quiet_NaN();
    double potentialShiftMargin = 1.0;

    // The QMC run writes CSV plus an interactive Plotly HTML file.
    bool writeOccupationMap = true;
    std::string outputPrefix = "qmc_occupation";

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
    double signSum = 0.0;
    double signedActiveSum = 0.0;
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
    double averageSign = 1.0;
    double signError = 0.0;
    double energy = 0.0;
    double energyError = 0.0;
    double cutoffHitFraction = 0.0;
    double diagonalAcceptance = 0.0;
    double wormAcceptance = 0.0;
    double stateAcceptance = 0.0;
    double potentialEnergy = 0.0;
    double potentialShiftSum = 0.0;
    int chainCount = 1;
    long long measurementCount = 0;
    double rawSignSum = 0.0;
    std::vector<double> occupations;
    std::vector<double> onsitePotential;
    std::vector<double> rawSignedOccupationSums;
};

// Basic utilities
int popcountMask(Mask x) {
    return __builtin_popcountll(x);
}

bool occupied(Mask state, int site) {
    return ((state >> site) & Mask{1}) != Mask{0};
}

Mask flipTwoSites(Mask state, int i, int j) {
    state ^= (Mask{1} << i);
    state ^= (Mask{1} << j);
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

std::string parseStringArg(
    const std::vector<std::string>& args,
    const std::string& name,
    const std::string& fallback
) {
    for (int i = 0; i + 1 < static_cast<int>(args.size()); ++i) {
        if (args[i] == name) {
            return args[i + 1];
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
std::vector<Bond> makeCubicLatticeBonds(int latticeSize, BoundaryCondition boundary) {
    std::vector<Bond> bonds;
    const int siteCount = latticeSize * latticeSize * latticeSize;
    const int expectedBondCount = (boundary == BoundaryCondition::Periodic)
        ? 3 * siteCount
        : 3 * latticeSize * latticeSize * (latticeSize - 1);
    bonds.reserve(std::max(0, expectedBondCount));

    auto site = [latticeSize](int x, int y, int z) {
        return x + latticeSize * (y + latticeSize * z);
    };

    for (int z = 0; z < latticeSize; ++z) {
        for (int y = 0; y < latticeSize; ++y) {
            for (int x = 0; x < latticeSize; ++x) {
                const int i = site(x, y, z);

                if (x + 1 < latticeSize) {
                    bonds.push_back({i, site(x + 1, y, z)});
                } else if (boundary == BoundaryCondition::Periodic) {
                    bonds.push_back({i, site(0, y, z)});
                }

                if (y + 1 < latticeSize) {
                    bonds.push_back({i, site(x, y + 1, z)});
                } else if (boundary == BoundaryCondition::Periodic) {
                    bonds.push_back({i, site(x, 0, z)});
                }

                if (z + 1 < latticeSize) {
                    bonds.push_back({i, site(x, y, z + 1)});
                } else if (boundary == BoundaryCondition::Periodic) {
                    bonds.push_back({i, site(x, y, 0)});
                }
            }
        }
    }

    return bonds;
}

void appendFixedParticleStates(
    int siteCount,
    int nextSite,
    int particlesRemaining,
    Mask current,
    std::vector<Mask>& basis
) {
    if (particlesRemaining == 0) {
        basis.push_back(current);
        return;
    }
    if (siteCount - nextSite < particlesRemaining) {
        return;
    }

    const int latestStart = siteCount - particlesRemaining;
    for (int site = nextSite; site <= latestStart; ++site) {
        appendFixedParticleStates(
            siteCount,
            site + 1,
            particlesRemaining - 1,
            current | (Mask{1} << site),
            basis
        );
    }
}

long double binomialEstimate(int n, int k) {
    k = std::min(k, n - k);
    long double value = 1.0L;
    for (int i = 1; i <= k; ++i) {
        value *= static_cast<long double>(n - k + i) / static_cast<long double>(i);
    }
    return value;
}

std::vector<Mask> makeFixedParticleBasis(int siteCount, int particleCount) {
    if (siteCount <= 0 || siteCount > 64) {
        throw std::runtime_error(
            "This uint64_t cubic-lattice implementation requires 1 <= L^3 <= 64."
        );
    }
    if (particleCount < 0 || particleCount > siteCount) {
        return {};
    }

    const long double estimatedSize = binomialEstimate(siteCount, particleCount);
    if (estimatedSize > 1000000.0L) {
        throw std::runtime_error(
            "The explicit fixed-N basis would exceed one million states. "
            "For a 4x4x4 lattice, use N <= 4 with this implementation."
        );
    }

    std::vector<Mask> basis;
    basis.reserve(static_cast<std::size_t>(estimatedSize));
    appendFixedParticleStates(siteCount, 0, particleCount, Mask{0}, basis);
    return basis;
}

std::vector<double> makeHarmonicPotential(const QmcParameters& params) {
    const int L = params.latticeSize;
    const int siteCount = L * L * L;
    std::vector<double> potential(siteCount, 0.0);

    if (!params.harmonicTrap) {
        return potential;
    }

    const double centerX = std::isnan(params.trapCenterX)
        ? 0.5 * static_cast<double>(L - 1)
        : params.trapCenterX;
    const double centerY = std::isnan(params.trapCenterY)
        ? 0.5 * static_cast<double>(L - 1)
        : params.trapCenterY;
    const double centerZ = std::isnan(params.trapCenterZ)
        ? 0.5 * static_cast<double>(L - 1)
        : params.trapCenterZ;

    for (int z = 0; z < L; ++z) {
        for (int y = 0; y < L; ++y) {
            for (int x = 0; x < L; ++x) {
                const int site = x + L * (y + L * z);
                const double dx = (static_cast<double>(x) - centerX) * params.latticeSpacing;
                const double dy = (static_cast<double>(y) - centerY) * params.latticeSpacing;
                const double dz = (static_cast<double>(z) - centerZ) * params.latticeSpacing;
                potential[site] = 0.5 * params.mass * params.omega * params.omega
                    * (dx * dx + dy * dy + dz * dz);
            }
        }
    }

    return potential;
}

std::vector<double> makePotentialShifts(
    const QmcParameters& params,
    const std::vector<double>& onsitePotential
) {
    std::vector<double> shifts(onsitePotential.size(), 0.0);
    if (!params.harmonicTrap) {
        return shifts;
    }

    const double margin = std::max(1e-12, params.potentialShiftMargin * params.hoppingT);
    for (int site = 0; site < static_cast<int>(onsitePotential.size()); ++site) {
        shifts[site] = std::max(0.0, onsitePotential[site]) + margin;
    }
    return shifts;
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
    Mask stateAfterAnnihilation = state ^ (Mask{1} << fromSite);
    signPower += countOccupiedBefore(stateAfterAnnihilation, toSite);
    return (signPower % 2 == 0) ? 1 : -1;
}

LocalMatrixElement applyPotentialOperator(
    Mask state,
    const BondOperator& op,
    const std::vector<double>& onsitePotential,
    const std::vector<double>& potentialShifts,
    double hoppingT
) {
    LocalMatrixElement out;
    if (op.site < 0 || op.site >= static_cast<int>(onsitePotential.size())) {
        return out;
    }

    const double ni = occupied(state, op.site) ? 1.0 : 0.0;
    const double physicalWeight = potentialShifts[op.site] - onsitePotential[op.site] * ni;
    const double scaledWeight = physicalWeight / hoppingT;

    if (scaledWeight <= 0.0) {
        return out;
    }

    out.valid = true;
    out.newState = state;
    out.absWeight = scaledWeight;
    out.sign = 1;
    return out;
}

LocalMatrixElement applyHardcoreBosonOperator(
    Mask state,
    const BondOperator& op,
    const std::vector<Bond>& bonds,
    const std::vector<double>& onsitePotential,
    const std::vector<double>& potentialShifts,
    double hoppingT
) {
    LocalMatrixElement out;

    if (op.kind == OperatorKind::Identity) {
        out.valid = true;
        out.newState = state;
        out.absWeight = 1.0;
        return out;
    }

    if (op.kind == OperatorKind::Potential) {
        return applyPotentialOperator(
            state,
            op,
            onsitePotential,
            potentialShifts,
            hoppingT
        );
    }

    const int i = bonds[op.bond].first;
    const int j = bonds[op.bond].second;
    const bool ni = occupied(state, i);
    const bool nj = occupied(state, j);

    if (op.kind == OperatorKind::Diagonal) {
        const double weight = 2.0 - static_cast<double>(ni) - static_cast<double>(nj);
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
    const std::vector<Bond>& bonds,
    const std::vector<double>& onsitePotential,
    const std::vector<double>& potentialShifts,
    double hoppingT
) {
    LocalMatrixElement out;

    if (op.kind == OperatorKind::Identity) {
        out.valid = true;
        out.newState = state;
        out.absWeight = 1.0;
        out.sign = 1;
        return out;
    }

    if (op.kind == OperatorKind::Potential) {
        return applyPotentialOperator(
            state,
            op,
            onsitePotential,
            potentialShifts,
            hoppingT
        );
    }

    const int i = bonds[op.bond].first;
    const int j = bonds[op.bond].second;
    const bool ni = occupied(state, i);
    const bool nj = occupied(state, j);

    if (op.kind == OperatorKind::Diagonal) {
        const double weight = 2.0 - static_cast<double>(ni) - static_cast<double>(nj);
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

        const int fromSite = ni ? i : j;
        const int toSite = ni ? j : i;

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
    ParticleModel model,
    const std::vector<double>& onsitePotential,
    const std::vector<double>& potentialShifts,
    double hoppingT
) {
    if (model == ParticleModel::HardcoreBoson) {
        return applyHardcoreBosonOperator(
            state, op, bonds, onsitePotential, potentialShifts, hoppingT
        );
    }
    return applySpinlessFermionOperator(
        state, op, bonds, onsitePotential, potentialShifts, hoppingT
    );
}

// Configuration weight
SignedLogWeight signedLogConfigurationWeight(
    Mask initialState,
    const std::vector<BondOperator>& operators,
    const std::vector<Bond>& bonds,
    double beta,
    double hoppingT,
    ParticleModel model,
    const std::vector<double>& onsitePotential,
    const std::vector<double>& potentialShifts
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
            model,
            onsitePotential,
            potentialShifts,
            hoppingT
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
    const std::vector<double>& onsitePotential,
    const std::vector<double>& potentialShifts,
    SignedLogWeight& currentWeight,
    AcceptanceCounter& counter,
    std::mt19937& rng
) {
    counter.attempts += 1;

    const int stringLength = static_cast<int>(operators.size());
    const int bondCount = static_cast<int>(bonds.size());
    const int siteCount = static_cast<int>(onsitePotential.size());
    const bool trapEnabled = std::any_of(
        potentialShifts.begin(), potentialShifts.end(),
        [](double value) { return value > 0.0; }
    );
    const int diagonalObjectCount = bondCount + (trapEnabled ? siteCount : 0);

    std::uniform_int_distribution<int> positionDist(0, stringLength - 1);
    std::uniform_int_distribution<int> objectDist(0, diagonalObjectCount - 1);

    const int position = positionDist(rng);
    const BondOperator oldOperator = operators[position];
    BondOperator newOperator = oldOperator;
    double proposalLogRatio = 0.0;

    if (oldOperator.kind == OperatorKind::Identity) {
        const int object = objectDist(rng);
        if (object < bondCount) {
            newOperator.kind = OperatorKind::Diagonal;
            newOperator.bond = object;
            newOperator.site = -1;
        } else {
            newOperator.kind = OperatorKind::Potential;
            newOperator.bond = -1;
            newOperator.site = object - bondCount;
        }
        proposalLogRatio = std::log(static_cast<double>(diagonalObjectCount));
    } else if (
        oldOperator.kind == OperatorKind::Diagonal ||
        oldOperator.kind == OperatorKind::Potential
    ) {
        newOperator.kind = OperatorKind::Identity;
        newOperator.bond = -1;
        newOperator.site = -1;
        proposalLogRatio = -std::log(static_cast<double>(diagonalObjectCount));
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
        onsitePotential,
        potentialShifts
    );

    const double logRatio =
        newWeight.logAbsWeight - currentWeight.logAbsWeight + proposalLogRatio;

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
    const std::vector<double>& onsitePotential,
    const std::vector<double>& potentialShifts,
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
        onsitePotential,
        potentialShifts
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
    const std::vector<double>& onsitePotential,
    const std::vector<double>& potentialShifts,
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
        if (operators[position].kind != OperatorKind::Identity) {
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
        onsitePotential,
        potentialShifts
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
    const std::vector<double>& onsitePotential,
    const std::vector<double>& potentialShifts,
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
        onsitePotential,
        potentialShifts
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
    const std::vector<double>& onsitePotential,
    const std::vector<double>& potentialShifts,
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
        onsitePotential,
        potentialShifts
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

std::vector<double> timeAveragedOccupations(
    Mask initialState,
    const std::vector<BondOperator>& operators,
    const std::vector<Bond>& bonds,
    ParticleModel model,
    const std::vector<double>& onsitePotential,
    const std::vector<double>& potentialShifts,
    double hoppingT,
    int siteCount
) {
    std::vector<double> occupation(siteCount, 0.0);
    Mask state = initialState;
    const double normalization = 1.0 / static_cast<double>(operators.size());

    for (const BondOperator& op : operators) {
        for (int site = 0; site < siteCount; ++site) {
            if (occupied(state, site)) {
                occupation[site] += normalization;
            }
        }

        if (op.kind == OperatorKind::Identity) {
            continue;
        }

        const LocalMatrixElement elem = applyModelOperator(
            state,
            op,
            bonds,
            model,
            onsitePotential,
            potentialShifts,
            hoppingT
        );
        if (!elem.valid) {
            throw std::runtime_error("Invalid operator encountered during occupation measurement.");
        }
        state = elem.newState;
    }

    // The original sampler does not explicitly enforce trace closure after
    // replaying the operator string, so the estimator follows its sampled
    // propagated path rather than rejecting an otherwise accepted state.
    return occupation;
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
        // Individual chains never write map files. Only the combined result is exported.
        localParams.writeOccupationMap = false;
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

    const int siteCount = params.latticeSize * params.latticeSize * params.latticeSize;
    const std::vector<Bond> bonds = makeCubicLatticeBonds(
        params.latticeSize,
        params.boundary
    );
    const std::vector<Mask> basis = makeFixedParticleBasis(
        siteCount,
        params.particleCount
    );
    const std::vector<double> onsitePotential = makeHarmonicPotential(params);
    const std::vector<double> potentialShifts = makePotentialShifts(
        params,
        onsitePotential
    );
    const double potentialShiftSum = std::accumulate(
        potentialShifts.begin(),
        potentialShifts.end(),
        0.0
    );

    if (basis.empty()) {
        throw std::runtime_error("Empty fixed-particle-number basis.");
    }

    std::mt19937 rng(params.seed);
    std::uniform_int_distribution<int> basisDist(0, static_cast<int>(basis.size()) - 1);

    Mask initialState = basis[basisDist(rng)];
    std::vector<BondOperator> operators(
        params.stringLength,
        BondOperator{OperatorKind::Identity, -1, -1}
    );

    SignedLogWeight currentWeight = signedLogConfigurationWeight(
        initialState,
        operators,
        bonds,
        params.beta,
        params.hoppingT,
        params.model,
        onsitePotential,
        potentialShifts
    );

    if (!currentWeight.valid) {
        throw std::runtime_error("Initial configuration has invalid weight.");
    }

    AcceptanceCounter diagonalCounter;
    AcceptanceCounter wormCounter;
    AcceptanceCounter stateCounter;

    const long long totalSweeps =
        params.thermalizationSweeps + params.measurementSweeps;
    const long long progressInterval = std::max<long long>(1, totalSweeps / 100);
    const long long totalMeasurements =
        (params.measurementSweeps + params.measureEvery - 1) / params.measureEvery;
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
    double signTotal = 0.0;
    double signedActiveTotal = 0.0;
    std::vector<double> signedOccupationTotal(siteCount, 0.0);

    const auto startTime = std::chrono::steady_clock::now();

    for (long long sweep = 0; sweep < totalSweeps; ++sweep) {
        if (params.showProgress && (sweep == 0 || sweep % progressInterval == 0)) {
            printProgressBar(
                sweep,
                totalSweeps,
                params.particleCount,
                startTime,
                "sweeps"
            );
        }

        if (
            params.sharedChainProgress != nullptr &&
            params.chainIndex >= 0 &&
            (sweep == 0 || sweep % progressInterval == 0)
        ) {
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
                onsitePotential,
                potentialShifts,
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
                onsitePotential,
                potentialShifts,
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
                onsitePotential,
                potentialShifts,
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
                onsitePotential,
                potentialShifts,
                currentWeight,
                stateCounter,
                rng
            );
        }

        if (
            sweep >= params.thermalizationSweeps &&
            ((sweep - params.thermalizationSweeps) % params.measureEvery == 0)
        ) {
            const int active = currentWeight.activeCount;
            int diagonal = 0;
            int hopping = 0;

            for (const BondOperator& op : operators) {
                if (op.kind == OperatorKind::Diagonal) {
                    diagonal += 1;
                } else if (op.kind == OperatorKind::Hopping) {
                    hopping += 1;
                }
            }

            const double sign = static_cast<double>(currentWeight.sign);
            const double signedActive = sign * static_cast<double>(active);
            const std::vector<double> occupationMeasurement = timeAveragedOccupations(
                initialState,
                operators,
                bonds,
                params.model,
                onsitePotential,
                potentialShifts,
                params.hoppingT,
                siteCount
            );

            activeTotal += active;
            diagonalTotal += diagonal;
            hoppingTotal += hopping;
            signTotal += sign;
            signedActiveTotal += signedActive;
            for (int site = 0; site < siteCount; ++site) {
                signedOccupationTotal[site] += sign * occupationMeasurement[site];
            }

            const int blockIndex = measurementBlockIndex(
                measurementIndex,
                totalMeasurements,
                static_cast<int>(blocks.size())
            );
            blocks[blockIndex].count += 1;
            blocks[blockIndex].activeSum += active;
            blocks[blockIndex].diagonalSum += diagonal;
            blocks[blockIndex].hoppingSum += hopping;
            blocks[blockIndex].signSum += sign;
            blocks[blockIndex].signedActiveSum += signedActive;

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
        printProgressBar(
            totalSweeps,
            totalSweeps,
            params.particleCount,
            startTime,
            "sweeps"
        );
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

        const double blockAverageSign =
            block.signSum / static_cast<double>(block.count);
        blockSigns.push_back(blockAverageSign);

        if (std::abs(block.signSum) < 1e-14) {
            continue;
        }
        blockEnergies.push_back(
            -block.signedActiveSum / (params.beta * block.signSum)
            + potentialShiftSum
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
    result.averageSign = signTotal / static_cast<double>(measurementIndex);
    result.signError = standardError(blockSigns);
    result.potentialShiftSum = potentialShiftSum;
    result.chainCount = 1;
    result.measurementCount = measurementIndex;
    result.rawSignSum = signTotal;
    result.rawSignedOccupationSums = signedOccupationTotal;
    result.onsitePotential = onsitePotential;
    result.occupations.assign(siteCount, std::numeric_limits<double>::quiet_NaN());

    if (std::abs(signTotal) < 1e-14) {
        result.energy = std::numeric_limits<double>::quiet_NaN();
        result.energyError = std::numeric_limits<double>::quiet_NaN();
    } else {
        result.energy =
            -signedActiveTotal / (params.beta * signTotal)
            + potentialShiftSum;
        result.energyError = standardError(blockEnergies);
        for (int site = 0; site < siteCount; ++site) {
            result.occupations[site] = signedOccupationTotal[site] / signTotal;
        }
    }

    result.potentialEnergy = 0.0;
    for (int site = 0; site < siteCount; ++site) {
        result.potentialEnergy += result.occupations[site] * onsitePotential[site];
    }

    result.cutoffHitFraction =
        static_cast<double>(cutoffHits) / static_cast<double>(measurementIndex);
    result.diagonalAcceptance = diagonalCounter.attempts == 0
        ? 0.0
        : static_cast<double>(diagonalCounter.accepted)
            / static_cast<double>(diagonalCounter.attempts);
    result.wormAcceptance = wormCounter.attempts == 0
        ? 0.0
        : static_cast<double>(wormCounter.accepted)
            / static_cast<double>(wormCounter.attempts);
    result.stateAcceptance = stateCounter.attempts == 0
        ? 0.0
        : static_cast<double>(stateCounter.accepted)
            / static_cast<double>(stateCounter.attempts);

    return result;
}

QmcResult combineChainResults(const std::vector<QmcResult>& results) {
    if (results.empty()) {
        throw std::runtime_error("No chain results to combine.");
    }

    QmcResult combined = results[0];
    const int chains = static_cast<int>(results.size());
    const int siteCount = combined.latticeSize * combined.latticeSize * combined.latticeSize;

    combined.averageActive = 0.0;
    combined.averageDiagonal = 0.0;
    combined.averageHopping = 0.0;
    combined.averageSign = 0.0;
    combined.signError = 0.0;
    combined.energy = 0.0;
    combined.diagonalAcceptance = 0.0;
    combined.wormAcceptance = 0.0;
    combined.stateAcceptance = 0.0;
    combined.cutoffHitFraction = 0.0;
    combined.potentialEnergy = 0.0;
    combined.chainCount = chains;
    combined.measurementCount = 0;
    combined.rawSignSum = 0.0;
    combined.rawSignedOccupationSums.assign(siteCount, 0.0);
    combined.occupations.assign(siteCount, std::numeric_limits<double>::quiet_NaN());

    std::vector<double> chainEnergies;
    std::vector<double> chainSigns;
    chainEnergies.reserve(chains);
    chainSigns.reserve(chains);

    for (const QmcResult& result : results) {
        if (result.latticeSize != combined.latticeSize ||
            result.particleCount != combined.particleCount ||
            result.occupations.size() != static_cast<std::size_t>(siteCount) ||
            result.rawSignedOccupationSums.size() != static_cast<std::size_t>(siteCount)) {
            throw std::runtime_error(
                "Cannot combine chains with incompatible lattice or occupation data."
            );
        }

        combined.averageActive += result.averageActive;
        combined.averageDiagonal += result.averageDiagonal;
        combined.averageHopping += result.averageHopping;
        combined.averageSign += result.averageSign;
        combined.energy += result.energy;
        combined.diagonalAcceptance += result.diagonalAcceptance;
        combined.wormAcceptance += result.wormAcceptance;
        combined.stateAcceptance += result.stateAcceptance;
        combined.cutoffHitFraction += result.cutoffHitFraction;
        combined.measurementCount += result.measurementCount;
        combined.rawSignSum += result.rawSignSum;

        for (int site = 0; site < siteCount; ++site) {
            combined.rawSignedOccupationSums[site] +=
                result.rawSignedOccupationSums[site];
        }

        chainEnergies.push_back(result.energy);
        chainSigns.push_back(result.averageSign);
    }

    combined.averageActive /= chains;
    combined.averageDiagonal /= chains;
    combined.averageHopping /= chains;
    combined.averageSign /= chains;
    combined.energy /= chains;
    combined.diagonalAcceptance /= chains;
    combined.wormAcceptance /= chains;
    combined.stateAcceptance /= chains;
    combined.cutoffHitFraction /= chains;

    if (std::abs(combined.rawSignSum) >= 1e-14) {
        for (int site = 0; site < siteCount; ++site) {
            combined.occupations[site] =
                combined.rawSignedOccupationSums[site] / combined.rawSignSum;
        }
    }

    combined.potentialEnergy = 0.0;
    for (int site = 0; site < siteCount; ++site) {
        combined.potentialEnergy +=
            combined.occupations[site] * combined.onsitePotential[site];
    }

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
              << std::setw(10) << result.averageSign
              << std::setw(10) << result.signError
              << std::setw(10) << result.diagonalAcceptance
              << std::setw(10) << result.wormAcceptance
              << std::setw(10) << result.stateAcceptance
              << "\n";
}

void printOccupationGrid(const QmcResult& result) {
    const int L = result.latticeSize;
    std::cout << "\nSite occupation slices <n_i> (one x-y plane per z):\n";
    std::cout << std::fixed << std::setprecision(6);
    for (int z = L - 1; z >= 0; --z) {
        std::cout << "z = " << z << "\n";
        for (int y = L - 1; y >= 0; --y) {
            for (int x = 0; x < L; ++x) {
                const int site = x + L * (y + L * z);
                std::cout << std::setw(11) << result.occupations[site];
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }
    std::cout << "sum = "
              << std::accumulate(result.occupations.begin(), result.occupations.end(), 0.0)
              << " (expected N=" << result.particleCount << ")\n";
}

void writeOccupationCsv(
    const QmcParameters& params,
    const QmcResult& result,
    const std::string& filename
) {
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Could not open CSV output: " + filename);
    }

    out << "site,x,y,z,physical_x,physical_y,physical_z,potential,occupation,random_particle_probability,chain_count,total_measurements\n";
    out << std::setprecision(12);
    const int L = result.latticeSize;
    const double centerX = std::isnan(params.trapCenterX)
        ? 0.5 * static_cast<double>(L - 1)
        : params.trapCenterX;
    const double centerY = std::isnan(params.trapCenterY)
        ? 0.5 * static_cast<double>(L - 1)
        : params.trapCenterY;
    const double centerZ = std::isnan(params.trapCenterZ)
        ? 0.5 * static_cast<double>(L - 1)
        : params.trapCenterZ;

    for (int z = 0; z < L; ++z) {
        for (int y = 0; y < L; ++y) {
            for (int x = 0; x < L; ++x) {
                const int site = x + L * (y + L * z);
                const double physicalX = (static_cast<double>(x) - centerX) * params.latticeSpacing;
                const double physicalY = (static_cast<double>(y) - centerY) * params.latticeSpacing;
                const double physicalZ = (static_cast<double>(z) - centerZ) * params.latticeSpacing;
                out << site << ',' << x << ',' << y << ',' << z << ','
                    << physicalX << ',' << physicalY << ',' << physicalZ << ','
                    << result.onsitePotential[site] << ','
                    << result.occupations[site] << ','
                    << result.occupations[site] / static_cast<double>(result.particleCount) << ','
                    << result.chainCount << ','
                    << result.measurementCount << '\n';
            }
        }
    }
}

void writeInteractive3dHtml(
    const QmcParameters& params,
    const QmcResult& result,
    const std::string& filename
) {
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Could not open HTML output: " + filename);
    }

    const int L = result.latticeSize;
    out << R"HTML(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QMC cubic lattice occupation map</title>
<script src="plotly.min.js"></script>
<style>
html,body{width:100%;height:100%;margin:0;background:#000;color:#eee;overflow:hidden;font-family:Inter,Segoe UI,Arial,sans-serif}
#plot{position:absolute;inset:0}
#panel{position:absolute;left:18px;top:16px;z-index:10;width:min(440px,calc(100vw - 70px));padding:14px 16px;border:1px solid rgba(255,190,40,.3);border-radius:12px;background:rgba(0,0,0,.74);box-shadow:0 0 34px rgba(255,120,0,.16);backdrop-filter:blur(8px)}
#panel h1{font-size:18px;margin:0 0 7px;color:#ffd166;text-shadow:0 0 14px rgba(255,197,66,.6)}
#panel p{font-size:12px;line-height:1.5;margin:4px 0;color:#bbb}
#panel b,#selected{color:#fff}
#error{display:none;position:absolute;inset:0;place-items:center;text-align:center;padding:40px;color:#ffbd59;background:#000;z-index:20}
</style>
</head>
<body>
<div id="plot"></div>
<div id="panel"><h1>QMC 3D cubic occupation</h1><p><b>Drag</b> to orbit, <b>scroll</b> to zoom, hover for values, and <b>click a site</b> to pin its details.</p><p id="meta"></p><p id="selected">No site selected.</p></div>
<div id="error">Plotly could not be loaded. Keep <b>plotly.min.js</b> in the same folder as this HTML file.</div>
<script>
)HTML";

    out << "const L=" << L << ";\n";
    out << "const N=" << result.particleCount << ";\n";
    out << "const beta=" << std::setprecision(15) << result.beta << ";\n";
    out << "const energy=" << result.energy << ";\n";
    out << "const chains=" << result.chainCount << ";\n";
    out << "const measurements=" << result.measurementCount << ";\n";
    out << "const boundary='" << boundaryName(params.boundary) << "';\n";
    out << "const model='" << modelName(params.model) << "';\n";
    out << "const trap=" << (params.harmonicTrap ? "true" : "false") << ";\n";
    out << "const spacing=" << params.latticeSpacing << ";\n";

    out << "const occupation=[";
    for (int i = 0; i < static_cast<int>(result.occupations.size()); ++i) {
        if (i) out << ',';
        out << result.occupations[i];
    }
    out << "];\nconst potential=[";
    for (int i = 0; i < static_cast<int>(result.onsitePotential.size()); ++i) {
        if (i) out << ',';
        out << result.onsitePotential[i];
    }
    out << "];\n";

    out << R"HTML(
if(typeof Plotly==='undefined'){
  document.getElementById('error').style.display='grid';
}else{
const xs=[],ys=[],zs=[],values=[],sizes=[],hover=[],siteIds=[];
const minP=Math.min(...occupation),maxP=Math.max(...occupation);
const span=Math.max(1e-15,maxP-minP);
for(let site=0;site<occupation.length;site++){
  const x=site%L;
  const y=Math.floor(site/L)%L;
  const z=Math.floor(site/(L*L));
  const p=occupation[site];
  const u=(p-minP)/span;
  xs.push(x);ys.push(y);zs.push(z);values.push(p);sizes.push(8+18*Math.pow(u,.72));siteIds.push(site);
  hover.push(`site ${site}<br>(x,y,z)=(${x},${y},${z})<br>&lt;n<sub>i</sub>&gt;=${p.toFixed(8)}<br>single-particle p=${(p/N).toFixed(8)}<br>V=${potential[site].toFixed(8)}`);
}
const edgeX=[],edgeY=[],edgeZ=[];
function edge(a,b){edgeX.push(a[0],b[0],null);edgeY.push(a[1],b[1],null);edgeZ.push(a[2],b[2],null);}
for(let z=0;z<L;z++)for(let y=0;y<L;y++)for(let x=0;x<L;x++){
  if(x+1<L)edge([x,y,z],[x+1,y,z]);
  if(y+1<L)edge([x,y,z],[x,y+1,z]);
  if(z+1<L)edge([x,y,z],[x,y,z+1]);
}
const traces=[
 {type:'scatter3d',mode:'lines',x:edgeX,y:edgeY,z:edgeZ,hoverinfo:'skip',showlegend:false,line:{color:'rgba(255,255,255,.10)',width:2}},
 {type:'scatter3d',mode:'markers',x:xs,y:ys,z:zs,hoverinfo:'skip',showlegend:false,
  marker:{size:sizes.map(v=>v*2.25),color:values,colorscale:'Inferno',cmin:minP,cmax:maxP,opacity:.13,line:{width:0}}},
 {type:'scatter3d',mode:'markers',x:xs,y:ys,z:zs,text:hover,customdata:siteIds,hovertemplate:'%{text}<extra></extra>',showlegend:false,
  marker:{size:sizes,color:values,colorscale:'Inferno',cmin:minP,cmax:maxP,opacity:1,line:{color:'rgba(255,235,145,.68)',width:1},
  colorbar:{title:{text:'&lt;n<sub>i</sub>&gt;',font:{color:'#eee'}},tickfont:{color:'#ddd'},thickness:14,len:.62,outlinecolor:'#555'}}},
 {type:'scatter3d',mode:'markers',x:[],y:[],z:[],hoverinfo:'skip',showlegend:false,
  marker:{size:34,color:'#fff3a1',opacity:.34,line:{color:'#fff6bd',width:2}}}
];
const axis={backgroundcolor:'#000',gridcolor:'rgba(255,255,255,.08)',zerolinecolor:'rgba(255,255,255,.16)',tickfont:{color:'#aaa'},titlefont:{color:'#ddd'},showbackground:true,dtick:1,range:[-.45,L-.55]};
const iso={eye:{x:1.55,y:1.55,z:1.45},up:{x:0,y:0,z:1}};
const layout={paper_bgcolor:'#000',plot_bgcolor:'#000',margin:{l:0,r:0,t:0,b:0},showlegend:false,
 scene:{bgcolor:'#000',xaxis:{...axis,title:'x'},yaxis:{...axis,title:'y'},zaxis:{...axis,title:'z'},aspectmode:'cube',camera:iso},
 updatemenus:[{type:'buttons',direction:'left',x:.99,y:.98,xanchor:'right',yanchor:'top',bgcolor:'rgba(20,20,20,.84)',bordercolor:'#6d4c1f',font:{color:'#ffd166'},buttons:[
  {label:'Isometric',method:'relayout',args:[{'scene.camera':iso}]},
  {label:'Top',method:'relayout',args:[{'scene.camera':{eye:{x:0,y:0,z:2.6},up:{x:0,y:1,z:0}}}]},
  {label:'Front',method:'relayout',args:[{'scene.camera':{eye:{x:0,y:2.6,z:0},up:{x:0,y:0,z:1}}}]},
  {label:'Side',method:'relayout',args:[{'scene.camera':{eye:{x:2.6,y:0,z:0},up:{x:0,y:0,z:1}}}]}
 ]}]
};
Plotly.newPlot('plot',traces,layout,{responsive:true,displaylogo:false,scrollZoom:true});
document.getElementById('meta').innerHTML=`${L}×${L}×${L} (${L**3} sites), N=${N}, β=${beta}, E=${energy.toFixed(8)}<br>${model}, ${boundary} boundaries, ${trap?'3D harmonic trap':'no external potential'}, a=${spacing}<br><b>${chains} chain${chains===1?'':'s'}</b>, ${measurements.toLocaleString()} pooled measurements`;
document.getElementById('plot').on('plotly_click',ev=>{
  const pt=ev.points?.[0]; if(!pt || pt.customdata===undefined)return;
  const site=Number(pt.customdata),x=site%L,y=Math.floor(site/L)%L,z=Math.floor(site/(L*L));
  Plotly.restyle('plot',{x:[[x]],y:[[y]],z:[[z]]},[3]);
  document.getElementById('selected').innerHTML=`Selected <b>site ${site}</b> at (${x}, ${y}, ${z})<br>&lt;n<sub>i</sub>&gt;=${occupation[site].toFixed(9)}, single-particle p=${(occupation[site]/N).toFixed(9)}, V=${potential[site].toFixed(9)}`;
});
}
</script>
</body>
</html>
)HTML";
}

void writePostAggregationMapOutputs(
    const QmcParameters& params,
    const QmcResult& aggregatedResult,
    const std::string& outputBase
) {
    if (aggregatedResult.occupations.empty()) {
        throw std::runtime_error(
            "Post-aggregation map export received no occupation data."
        );
    }

    const std::string csvFile = outputBase + ".csv";
    const std::string htmlFile = outputBase + ".html";

    writeOccupationCsv(params, aggregatedResult, csvFile);
    writeInteractive3dHtml(params, aggregatedResult, htmlFile);

    std::cout << "Wrote one post-aggregation map from "
              << aggregatedResult.chainCount << " chain"
              << (aggregatedResult.chainCount == 1 ? "" : "s")
              << " and " << aggregatedResult.measurementCount
              << " pooled measurements:\n";
    std::cout << "  " << csvFile << "\n";
    std::cout << "  " << htmlFile << "\n";
}

void printHelp() {
    std::cout << "Hard-core boson / spinless fermion QMC on a 3D cubic lattice\n\n";
    std::cout << "Compile:\n";
    std::cout << "  g++ -std=c++17 -O3 -fopenmp qmc_sho_cubic_3d.cpp -o qmc_sho_cubic_3d\n\n";
    std::cout << "Example:\n";
    std::cout << "  ./qmc_sho_cubic_3d --model boson --boundary periodic --L 4 --N 2 --trap "
                 "--mass 1 --omega 0.5 --a 1 --M 800 --chains 12 --output periodic_sho_cubic\n\n";
    std::cout << "Core options:\n";
    std::cout << "  --model <boson|fermion>       particle model, default boson\n";
    std::cout << "  --boundary <periodic|open>    default periodic\n";
    std::cout << "  --N <int>                     fixed particle number\n";
    std::cout << "  --L <int>                     cubic lattice side length (L=4 gives 64 sites)\n";
    std::cout << "  --beta <float>                inverse temperature\n";
    std::cout << "  --t <float>                   hopping energy\n";
    std::cout << "  --M <int>                     SSE operator-string cutoff, default 800\n";
    std::cout << "  --therm <int>                 thermalization sweeps\n";
    std::cout << "  --sweeps <int>                measurement sweeps\n";
    std::cout << "  --measure_every <int>         measurement interval\n";
    std::cout << "  --chains <int>                independent OpenMP chains\n\n";
    std::cout << "Harmonic potential:\n";
    std::cout << "  --trap                        enable V=1/2 m omega^2 (x^2+y^2+z^2)\n";
    std::cout << "  --mass <float>                default 1\n";
    std::cout << "  --omega <float>               default 0.5\n";
    std::cout << "  --a <float>                   spatial lattice spacing\n";
    std::cout << "  --center_x <float>            default (L-1)/2\n";
    std::cout << "  --center_y <float>            default (L-1)/2\n";
    std::cout << "  --center_z <float>            default (L-1)/2\n";
    std::cout << "  --shift_margin <float>        positive SSE shift in units of t, default 1\n\n";
    std::cout << "Map output:\n";
    std::cout << "  --output <prefix>             one post-aggregation HTML/CSV prefix\n";
    std::cout << "  --no_map                      skip HTML and CSV output\n";
    std::cout << "  With --chains K, all K chains are pooled before one map is written.\n";
    std::cout << "  Keep plotly.min.js beside the generated HTML file.\n";
    std::cout << "  --seed <int>                  random seed\n";
    std::cout << "  --random_seed                 use std::random_device\n";
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
    params.stringLength = parseIntArg(args, "--M", params.stringLength);
    params.thermalizationSweeps = parseLongLongArg(args, "--therm", params.thermalizationSweeps);
    params.measurementSweeps = parseLongLongArg(args, "--sweeps", params.measurementSweeps);
    params.measureEvery = parseIntArg(args, "--measure_every", params.measureEvery);
    params.blockCount = parseIntArg(args, "--blocks", params.blockCount);
    params.seed = parseUintArg(args, "--seed", params.seed);
    params.harmonicTrap = hasFlag(args, "--trap");
    params.mass = parseDoubleArg(args, "--mass", params.mass);
    params.omega = parseDoubleArg(args, "--omega", params.omega);
    params.latticeSpacing = parseDoubleArg(args, "--a", params.latticeSpacing);
    params.trapCenterX = parseDoubleArg(args, "--center_x", params.trapCenterX);
    params.trapCenterY = parseDoubleArg(args, "--center_y", params.trapCenterY);
    params.trapCenterZ = parseDoubleArg(args, "--center_z", params.trapCenterZ);
    params.potentialShiftMargin = parseDoubleArg(
        args,
        "--shift_margin",
        params.potentialShiftMargin
    );
    params.outputPrefix = parseStringArg(args, "--output", params.outputPrefix);
    params.writeOccupationMap = !hasFlag(args, "--no_map");

    if (hasFlag(args, "--random_seed")) {
        std::random_device rd;
        params.seed = rd();
    }

    int chains = parseIntArg(args, "--chains", 1);

    std::vector<int> particleNumbers;
    if (hasFlag(args, "--all")) {
        throw std::runtime_error(
            "--all is disabled for the 64-site cubic lattice because the explicit fixed-N basis grows combinatorially. Run N=2, N=3, or N=4 separately."
        );
    }
    particleNumbers = {params.particleCount};


    std::cout << "Stochastic worm QMC sampler on an L x L x L cubic lattice\n";
    std::cout << "model=" << modelName(params.model)
              << ", boundary=" << boundaryName(params.boundary)
              << ", lattice=" << params.latticeSize << "x" << params.latticeSize << "x" << params.latticeSize
              << ", beta=" << params.beta
              << ", t=" << params.hoppingT
              << ", trap=" << (params.harmonicTrap ? "sho" : "off")
              << ", a=" << params.latticeSpacing
              << ", M=" << params.stringLength
              << ", therm=" << params.thermalizationSweeps
              << ", sweeps=" << params.measurementSweeps
              << ", chains=" << chains
              << ", seed=" << params.seed
              << "\n\n";

    std::cout << std::left << std::setw(4) << "N"
              << std::right << std::setw(14) << "QMC E"
              << " +/- " << std::setw(9) << "err"
              << std::setw(14) << "Exact"
              << std::setw(14) << "diff"
              << std::setw(10) << "<Nop>"
              << std::setw(10) << "<D>"
              << std::setw(10) << "<H>"
              << std::setw(10) << "<sgn>"
              << std::setw(10) << "sgn_err"
              << std::setw(10) << "accD"
              << std::setw(10) << "accW"
              << std::setw(10) << "accS"
              << "\n";

    std::cout << std::string(149, '-') << "\n";

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

        double exactValue = std::numeric_limits<double>::quiet_NaN();

        printResultRow(result, exactValue);
        printOccupationGrid(result);
        std::cout << "<V> = " << result.potentialEnergy << "\n";

        if (params.writeOccupationMap) {
            std::string outputBase = params.outputPrefix;
            if (particleNumbers.size() > 1) {
                outputBase += "_N" + std::to_string(particleCount);
            }

            // This is deliberately after runParallelChains() returns. At this point
            // `result` contains the pooled, sign-reweighted occupation estimator
            // from every chain, so exactly one map is written.
            writePostAggregationMapOutputs(params, result, outputBase);
        }

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
