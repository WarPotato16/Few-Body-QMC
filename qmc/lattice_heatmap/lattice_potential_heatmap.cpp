#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using Mask = std::uint64_t;
using Bond = std::pair<int, int>;
using Matrix = std::vector<std::vector<double>>;

constexpr double pi = 3.14159265358979323846;

enum class BoundaryCondition { Periodic, Open };
enum class ParticleModel { HardcoreBoson, SpinlessFermion };
enum class PotentialType {
    None,
    Harmonic,
    Quartic,
    GaussianWell,
    GaussianBarrier,
    DoubleWellX,
    LinearX,
    Checkerboard,
    Disorder,
    Impurity
};

struct Parameters {
    int latticeSize = 4;
    int particleCount = 2;
    double hoppingT = 1.0;
    BoundaryCondition boundary = BoundaryCondition::Periodic;
    ParticleModel model = ParticleModel::HardcoreBoson;
    PotentialType potential = PotentialType::Harmonic;

    double mass = 1.0;
    double omega = 0.5;
    double latticeSpacing = 1.0;
    double centerX = std::numeric_limits<double>::quiet_NaN();
    double centerY = std::numeric_limits<double>::quiet_NaN();

    // Generic controls used by the non-harmonic potentials.
    double strength = 1.0;
    double sigma = 1.0;
    double separation = 1.0;
    double field = 0.25;
    double disorderWidth = 1.0;
    int impurityX = 0;
    int impurityY = 0;
    unsigned int seed = 12345;

    std::string outputPrefix = "lattice_ground_state";
    int jacobiSweeps = 100;
    double jacobiTolerance = 1e-12;
};

struct EigenSystem {
    std::vector<double> eigenvalues;
    Matrix eigenvectors; // Eigenvectors are stored as columns.
};

int popcountMask(Mask state) {
#if defined(__GNUG__) || defined(__clang__)
    return __builtin_popcountll(state);
#else
    int count = 0;
    while (state != 0) {
        state &= state - 1;
        ++count;
    }
    return count;
#endif
}

bool occupied(Mask state, int site) {
    return ((state >> site) & Mask{1}) != 0;
}

Mask flipTwoSites(Mask state, int i, int j) {
    state ^= (Mask{1} << i);
    state ^= (Mask{1} << j);
    return state;
}

std::string boundaryName(BoundaryCondition boundary) {
    return boundary == BoundaryCondition::Periodic ? "periodic" : "open";
}

std::string modelName(ParticleModel model) {
    return model == ParticleModel::HardcoreBoson ? "hardcore_boson" : "spinless_fermion";
}

std::string potentialName(PotentialType potential) {
    switch (potential) {
        case PotentialType::None: return "none";
        case PotentialType::Harmonic: return "sho";
        case PotentialType::Quartic: return "quartic";
        case PotentialType::GaussianWell: return "gaussian_well";
        case PotentialType::GaussianBarrier: return "gaussian_barrier";
        case PotentialType::DoubleWellX: return "double_well_x";
        case PotentialType::LinearX: return "linear_x";
        case PotentialType::Checkerboard: return "checkerboard";
        case PotentialType::Disorder: return "disorder";
        case PotentialType::Impurity: return "impurity";
    }
    throw std::runtime_error("Unknown potential type.");
}

PotentialType parsePotential(const std::string& value) {
    if (value == "none") return PotentialType::None;
    if (value == "sho" || value == "harmonic") return PotentialType::Harmonic;
    if (value == "quartic") return PotentialType::Quartic;
    if (value == "gaussian_well") return PotentialType::GaussianWell;
    if (value == "gaussian_barrier") return PotentialType::GaussianBarrier;
    if (value == "double_well" || value == "double_well_x") return PotentialType::DoubleWellX;
    if (value == "linear" || value == "linear_x") return PotentialType::LinearX;
    if (value == "checkerboard" || value == "staggered") return PotentialType::Checkerboard;
    if (value == "disorder") return PotentialType::Disorder;
    if (value == "impurity") return PotentialType::Impurity;
    throw std::runtime_error("Unknown --potential value: " + value);
}

std::vector<Bond> makeSquareLatticeBonds(int latticeSize, BoundaryCondition boundary) {
    std::vector<Bond> bonds;
    const int expected = boundary == BoundaryCondition::Periodic
        ? 2 * latticeSize * latticeSize
        : 2 * latticeSize * (latticeSize - 1);
    bonds.reserve(std::max(0, expected));

    auto site = [latticeSize](int x, int y) {
        return y * latticeSize + x;
    };

    for (int y = 0; y < latticeSize; ++y) {
        for (int x = 0; x < latticeSize; ++x) {
            const int i = site(x, y);

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
    if (siteCount <= 0 || siteCount > 64) {
        throw std::runtime_error("This program requires 1 <= L^2 <= 64.");
    }
    if (particleCount < 0 || particleCount > siteCount) {
        throw std::runtime_error("Particle count must be between 0 and L^2.");
    }

    std::vector<Mask> basis;
    if (particleCount == 0) {
        basis.push_back(0);
        return basis;
    }
    if (particleCount == siteCount) {
        basis.push_back((Mask{1} << siteCount) - 1);
        return basis;
    }

    Mask state = (Mask{1} << particleCount) - 1;
    const Mask limit = Mask{1} << siteCount;

    while (state < limit) {
        basis.push_back(state);

        // Gosper's hack: next bit mask with the same population count.
        const Mask smallest = state & (~state + 1);
        const Mask ripple = state + smallest;
        const Mask ones = ((state ^ ripple) >> 2) / smallest;
        state = ripple | ones;
    }

    return basis;
}

int countOccupiedBefore(Mask state, int site) {
    if (site <= 0) return 0;
    const Mask lowerMask = (Mask{1} << site) - 1;
    return popcountMask(state & lowerMask);
}

int fermionHopSign(Mask state, int fromSite, int toSite) {
    int exponent = countOccupiedBefore(state, fromSite);
    const Mask afterAnnihilation = state ^ (Mask{1} << fromSite);
    exponent += countOccupiedBefore(afterAnnihilation, toSite);
    return exponent % 2 == 0 ? 1 : -1;
}

std::vector<double> makeOnsitePotential(const Parameters& params) {
    const int L = params.latticeSize;
    const int siteCount = L * L;
    const double centerX = std::isnan(params.centerX) ? 0.5 * (L - 1) : params.centerX;
    const double centerY = std::isnan(params.centerY) ? 0.5 * (L - 1) : params.centerY;

    std::vector<double> potential(siteCount, 0.0);
    std::mt19937 rng(params.seed);
    std::uniform_real_distribution<double> disorder(
        -0.5 * params.disorderWidth,
        0.5 * params.disorderWidth
    );

    for (int y = 0; y < L; ++y) {
        for (int x = 0; x < L; ++x) {
            const int site = y * L + x;
            const double dx = (static_cast<double>(x) - centerX) * params.latticeSpacing;
            const double dy = (static_cast<double>(y) - centerY) * params.latticeSpacing;
            const double r2 = dx * dx + dy * dy;

            switch (params.potential) {
                case PotentialType::None:
                    potential[site] = 0.0;
                    break;
                case PotentialType::Harmonic:
                    potential[site] = 0.5 * params.mass * params.omega * params.omega * r2;
                    break;
                case PotentialType::Quartic:
                    potential[site] = params.strength * r2 * r2;
                    break;
                case PotentialType::GaussianWell:
                    potential[site] = -params.strength
                        * std::exp(-r2 / (2.0 * params.sigma * params.sigma));
                    break;
                case PotentialType::GaussianBarrier:
                    potential[site] = params.strength
                        * std::exp(-r2 / (2.0 * params.sigma * params.sigma));
                    break;
                case PotentialType::DoubleWellX: {
                    const double b2 = params.separation * params.separation;
                    const double longitudinal = params.strength * (dx * dx - b2) * (dx * dx - b2);
                    const double transverse = 0.5 * params.mass * params.omega * params.omega * dy * dy;
                    potential[site] = longitudinal + transverse;
                    break;
                }
                case PotentialType::LinearX:
                    potential[site] = params.field * dx;
                    break;
                case PotentialType::Checkerboard:
                    potential[site] = ((x + y) % 2 == 0 ? -params.strength : params.strength);
                    break;
                case PotentialType::Disorder:
                    potential[site] = disorder(rng);
                    break;
                case PotentialType::Impurity:
                    potential[site] = (x == params.impurityX && y == params.impurityY)
                        ? params.strength
                        : 0.0;
                    break;
            }
        }
    }

    return potential;
}

Matrix makeHamiltonian(
    const Parameters& params,
    const std::vector<Mask>& basis,
    const std::vector<Bond>& bonds,
    const std::vector<double>& onsitePotential
) {
    const int dimension = static_cast<int>(basis.size());
    const int siteCount = params.latticeSize * params.latticeSize;
    Matrix hamiltonian(dimension, std::vector<double>(dimension, 0.0));

    std::unordered_map<Mask, int> basisIndex;
    basisIndex.reserve(basis.size() * 2);
    for (int row = 0; row < dimension; ++row) {
        basisIndex[basis[row]] = row;
    }

    for (int row = 0; row < dimension; ++row) {
        const Mask state = basis[row];
        double diagonalEnergy = 0.0;

        // Same original diagonal bond term as the QMC code:
        // -t * sum_<ij> (2 - n_i - n_j).
        for (const Bond& bond : bonds) {
            const int i = bond.first;
            const int j = bond.second;
            const bool ni = occupied(state, i);
            const bool nj = occupied(state, j);

            diagonalEnergy += -params.hoppingT
                * (2.0 - static_cast<double>(ni) - static_cast<double>(nj));

            if (ni != nj) {
                const Mask newState = flipTwoSites(state, i, j);
                const auto found = basisIndex.find(newState);
                if (found == basisIndex.end()) {
                    throw std::runtime_error("Internal basis lookup failure.");
                }

                int sign = 1;
                if (params.model == ParticleModel::SpinlessFermion) {
                    const int fromSite = ni ? i : j;
                    const int toSite = ni ? j : i;
                    sign = fermionHopSign(state, fromSite, toSite);
                }

                hamiltonian[row][found->second] += -params.hoppingT * static_cast<double>(sign);
            }
        }

        for (int site = 0; site < siteCount; ++site) {
            if (occupied(state, site)) {
                diagonalEnergy += onsitePotential[site];
            }
        }

        hamiltonian[row][row] += diagonalEnergy;
    }

    // Numerical and implementation sanity check.
    for (int i = 0; i < dimension; ++i) {
        for (int j = i + 1; j < dimension; ++j) {
            if (std::abs(hamiltonian[i][j] - hamiltonian[j][i]) > 1e-10) {
                throw std::runtime_error("Hamiltonian is not symmetric; check hopping signs and bonds.");
            }
        }
    }

    return hamiltonian;
}

EigenSystem jacobiDiagonalize(Matrix matrix, int maxSweeps, double tolerance) {
    const int n = static_cast<int>(matrix.size());
    Matrix eigenvectors(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        eigenvectors[i][i] = 1.0;
    }

    for (int sweep = 0; sweep < maxSweeps; ++sweep) {
        double offDiagonalNormSquared = 0.0;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                offDiagonalNormSquared += 2.0 * matrix[p][q] * matrix[p][q];
            }
        }

        const double offDiagonalNorm = std::sqrt(offDiagonalNormSquared);
        if (offDiagonalNorm < tolerance) {
            std::vector<double> eigenvalues(n);
            for (int i = 0; i < n; ++i) eigenvalues[i] = matrix[i][i];
            return EigenSystem{eigenvalues, eigenvectors};
        }

        const double threshold = sweep < 4
            ? 0.2 * offDiagonalNorm / static_cast<double>(n * n)
            : 0.0;

        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = matrix[p][q];
                if (std::abs(apq) <= threshold) continue;

                const double app = matrix[p][p];
                const double aqq = matrix[q][q];
                const double theta = (aqq - app) / (2.0 * apq);
                double tangent = 1.0 / (std::abs(theta) + std::sqrt(1.0 + theta * theta));
                if (theta < 0.0) tangent = -tangent;

                const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
                const double sine = tangent * cosine;

                for (int k = 0; k < n; ++k) {
                    if (k == p || k == q) continue;
                    const double akp = matrix[k][p];
                    const double akq = matrix[k][q];
                    const double newKp = cosine * akp - sine * akq;
                    const double newKq = sine * akp + cosine * akq;
                    matrix[k][p] = matrix[p][k] = newKp;
                    matrix[k][q] = matrix[q][k] = newKq;
                }

                matrix[p][p] = cosine * cosine * app
                    - 2.0 * sine * cosine * apq
                    + sine * sine * aqq;
                matrix[q][q] = sine * sine * app
                    + 2.0 * sine * cosine * apq
                    + cosine * cosine * aqq;
                matrix[p][q] = matrix[q][p] = 0.0;

                for (int k = 0; k < n; ++k) {
                    const double vkp = eigenvectors[k][p];
                    const double vkq = eigenvectors[k][q];
                    eigenvectors[k][p] = cosine * vkp - sine * vkq;
                    eigenvectors[k][q] = sine * vkp + cosine * vkq;
                }
            }
        }
    }

    throw std::runtime_error(
        "Jacobi diagonalization did not converge. Increase --jacobi_sweeps or relax --jacobi_tol."
    );
}

std::vector<double> groundStateVector(const EigenSystem& system, double& groundEnergy) {
    const auto minimum = std::min_element(system.eigenvalues.begin(), system.eigenvalues.end());
    const int column = static_cast<int>(std::distance(system.eigenvalues.begin(), minimum));
    groundEnergy = *minimum;

    std::vector<double> vector(system.eigenvectors.size(), 0.0);
    for (int row = 0; row < static_cast<int>(vector.size()); ++row) {
        vector[row] = system.eigenvectors[row][column];
    }

    double norm = 0.0;
    for (double value : vector) norm += value * value;
    norm = std::sqrt(norm);
    if (norm <= 0.0) throw std::runtime_error("Ground-state eigenvector has zero norm.");
    for (double& value : vector) value /= norm;
    return vector;
}

std::vector<double> calculateOccupations(
    const std::vector<Mask>& basis,
    const std::vector<double>& groundState,
    int siteCount
) {
    std::vector<double> occupations(siteCount, 0.0);

    for (int alpha = 0; alpha < static_cast<int>(basis.size()); ++alpha) {
        const double probability = groundState[alpha] * groundState[alpha];
        for (int site = 0; site < siteCount; ++site) {
            if (occupied(basis[alpha], site)) {
                occupations[site] += probability;
            }
        }
    }

    return occupations;
}

double calculateBondExpectation(
    const Parameters& params,
    const std::vector<Mask>& basis,
    const std::vector<Bond>& bonds,
    const std::vector<double>& groundState
) {
    double expectation = 0.0;
    for (int alpha = 0; alpha < static_cast<int>(basis.size()); ++alpha) {
        double energy = 0.0;
        for (const Bond& bond : bonds) {
            const double ni = occupied(basis[alpha], bond.first) ? 1.0 : 0.0;
            const double nj = occupied(basis[alpha], bond.second) ? 1.0 : 0.0;
            energy += -params.hoppingT * (2.0 - ni - nj);
        }
        expectation += groundState[alpha] * groundState[alpha] * energy;
    }
    return expectation;
}

double calculatePotentialExpectation(
    const std::vector<double>& occupations,
    const std::vector<double>& onsitePotential
) {
    double expectation = 0.0;
    for (int site = 0; site < static_cast<int>(occupations.size()); ++site) {
        expectation += occupations[site] * onsitePotential[site];
    }
    return expectation;
}

std::string rgbColor(double normalized) {
    normalized = std::max(0.0, std::min(1.0, normalized));
    // A simple light-to-dark blue map suitable for printed grids.
    const int r = static_cast<int>(std::round(242.0 + normalized * (20.0 - 242.0)));
    const int g = static_cast<int>(std::round(247.0 + normalized * (89.0 - 247.0)));
    const int b = static_cast<int>(std::round(252.0 + normalized * (140.0 - 252.0)));
    return "rgb(" + std::to_string(r) + "," + std::to_string(g) + "," + std::to_string(b) + ")";
}

void writeCsv(
    const std::string& filename,
    const Parameters& params,
    const std::vector<double>& potential,
    const std::vector<double>& occupations
) {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Could not open " + filename);

    out << "site,x,y,potential,occupation,random_particle_probability\n";
    out << std::setprecision(12);
    for (int y = 0; y < params.latticeSize; ++y) {
        for (int x = 0; x < params.latticeSize; ++x) {
            const int site = y * params.latticeSize + x;
            out << site << ',' << x << ',' << y << ','
                << potential[site] << ',' << occupations[site] << ','
                << occupations[site] / static_cast<double>(params.particleCount) << '\n';
        }
    }
}

void writeSvgHeatmap(
    const std::string& filename,
    const Parameters& params,
    const std::vector<double>& potential,
    const std::vector<double>& occupations,
    double groundEnergy
) {
    const int L = params.latticeSize;
    const int cell = 105;
    const int grid = L * cell;
    const int margin = 60;
    const int panelGap = 90;
    const int titleHeight = 100;
    const int legendHeight = 75;
    const int width = 2 * grid + 2 * margin + panelGap;
    const int height = titleHeight + grid + legendHeight + margin;

    const auto [potentialMinIt, potentialMaxIt] = std::minmax_element(potential.begin(), potential.end());
    const auto [occupationMinIt, occupationMaxIt] = std::minmax_element(occupations.begin(), occupations.end());
    const double potentialMin = *potentialMinIt;
    const double potentialMax = *potentialMaxIt;
    const double occupationMin = *occupationMinIt;
    const double occupationMax = *occupationMaxIt;

    auto normalize = [](double value, double low, double high) {
        if (std::abs(high - low) < 1e-15) return 0.5;
        return (value - low) / (high - low);
    };

    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Could not open " + filename);

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << ' ' << height << "\">\n";
    out << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    out << "<style>text{font-family:Arial,Helvetica,sans-serif;} .title{font-size:25px;font-weight:700;}"
           " .subtitle{font-size:15px;fill:#333;} .panel{font-size:20px;font-weight:700;}"
           " .value{font-size:16px;font-weight:700;} .site{font-size:12px;}"
           " .axis{font-size:13px;fill:#333;}</style>\n";

    out << "<text x=\"" << width / 2 << "\" y=\"34\" text-anchor=\"middle\" class=\"title\">"
        << "Ground-state lattice heatmap</text>\n";
    out << "<text x=\"" << width / 2 << "\" y=\"62\" text-anchor=\"middle\" class=\"subtitle\">"
        << "L=" << L << ", N=" << params.particleCount
        << ", model=" << modelName(params.model)
        << ", boundary=" << boundaryName(params.boundary)
        << ", potential=" << potentialName(params.potential)
        << ", E0=" << std::fixed << std::setprecision(8) << groundEnergy
        << "</text>\n";

    const int potentialX = margin;
    const int occupationX = margin + grid + panelGap;
    const int gridY = titleHeight;

    out << "<text x=\"" << potentialX + grid / 2 << "\" y=\"" << gridY - 20
        << "\" text-anchor=\"middle\" class=\"panel\">Onsite potential V(x,y)</text>\n";
    out << "<text x=\"" << occupationX + grid / 2 << "\" y=\"" << gridY - 20
        << "\" text-anchor=\"middle\" class=\"panel\">Occupation probability &lt;n_i&gt;</text>\n";

    for (int displayRow = 0; displayRow < L; ++displayRow) {
        const int y = L - 1 - displayRow;
        for (int x = 0; x < L; ++x) {
            const int site = y * L + x;
            const int py = gridY + displayRow * cell;

            const double pn = normalize(potential[site], potentialMin, potentialMax);
            const double on = normalize(occupations[site], occupationMin, occupationMax);
            const std::string potentialTextColor = pn > 0.55 ? "white" : "#111";
            const std::string occupationTextColor = on > 0.55 ? "white" : "#111";

            const int pxPotential = potentialX + x * cell;
            out << "<rect x=\"" << pxPotential << "\" y=\"" << py << "\" width=\"" << cell
                << "\" height=\"" << cell << "\" fill=\"" << rgbColor(pn)
                << "\" stroke=\"#222\" stroke-width=\"1.5\"/>\n";
            out << "<text x=\"" << pxPotential + cell / 2 << "\" y=\"" << py + cell / 2 - 2
                << "\" text-anchor=\"middle\" fill=\"" << potentialTextColor << "\" class=\"value\">"
                << std::fixed << std::setprecision(4) << potential[site] << "</text>\n";
            out << "<text x=\"" << pxPotential + cell / 2 << "\" y=\"" << py + cell / 2 + 22
                << "\" text-anchor=\"middle\" fill=\"" << potentialTextColor << "\" class=\"site\">site "
                << site << " (" << x << ',' << y << ")</text>\n";

            const int pxOccupation = occupationX + x * cell;
            out << "<rect x=\"" << pxOccupation << "\" y=\"" << py << "\" width=\"" << cell
                << "\" height=\"" << cell << "\" fill=\"" << rgbColor(on)
                << "\" stroke=\"#222\" stroke-width=\"1.5\"/>\n";
            out << "<text x=\"" << pxOccupation + cell / 2 << "\" y=\"" << py + cell / 2 - 2
                << "\" text-anchor=\"middle\" fill=\"" << occupationTextColor << "\" class=\"value\">"
                << std::fixed << std::setprecision(6) << occupations[site] << "</text>\n";
            out << "<text x=\"" << pxOccupation + cell / 2 << "\" y=\"" << py + cell / 2 + 22
                << "\" text-anchor=\"middle\" fill=\"" << occupationTextColor << "\" class=\"site\">site "
                << site << " (" << x << ',' << y << ")</text>\n";
        }
    }

    const int legendY = gridY + grid + 28;
    out << "<text x=\"" << potentialX << "\" y=\"" << legendY
        << "\" class=\"axis\">min " << std::fixed << std::setprecision(5) << potentialMin
        << "   max " << potentialMax << "</text>\n";
    out << "<text x=\"" << occupationX << "\" y=\"" << legendY
        << "\" class=\"axis\">min " << std::fixed << std::setprecision(6) << occupationMin
        << "   max " << occupationMax << "   sum "
        << std::accumulate(occupations.begin(), occupations.end(), 0.0) << "</text>\n";

    out << "</svg>\n";
}

void printGrid(const std::vector<double>& values, int L, int precision, const std::string& title) {
    std::cout << title << "\n";
    std::cout << std::fixed << std::setprecision(precision);
    for (int y = L - 1; y >= 0; --y) {
        for (int x = 0; x < L; ++x) {
            std::cout << std::setw(12) << values[y * L + x];
        }
        std::cout << '\n';
    }
}

std::string getArgument(const std::vector<std::string>& args, const std::string& name, const std::string& fallback) {
    for (int i = 0; i + 1 < static_cast<int>(args.size()); ++i) {
        if (args[i] == name) return args[i + 1];
    }
    return fallback;
}

bool hasFlag(const std::vector<std::string>& args, const std::string& name) {
    return std::find(args.begin(), args.end(), name) != args.end();
}

void printHelp() {
    std::cout
        << "Exact ground-state occupation heatmap for the lattice QMC Hamiltonian\n\n"
        << "Build:\n"
        << "  g++ -std=c++17 -O3 lattice_potential_heatmap.cpp -o lattice_heatmap\n\n"
        << "Example matching the periodic 4x4, N=2 SHO calculation:\n"
        << "  ./lattice_heatmap --L 4 --N 2 --boundary periodic --potential sho \\\n\n      --mass 1 --omega 0.5 --t 1 --output periodic_sho\n\n"
        << "Core options:\n"
        << "  --L <int>                 lattice side length (default 4)\n"
        << "  --N <int>                 fixed particle number (default 2)\n"
        << "  --t <float>               hopping scale (default 1)\n"
        << "  --boundary periodic|open  default periodic\n"
        << "  --model boson|fermion     default boson\n"
        << "  --output <prefix>         SVG/CSV output prefix\n\n"
        << "Potential options:\n"
        << "  --potential none|sho|quartic|gaussian_well|gaussian_barrier\n"
        << "              |double_well_x|linear_x|checkerboard|disorder|impurity\n"
        << "  --mass <float>            SHO mass and transverse double-well mass\n"
        << "  --omega <float>           SHO frequency\n"
        << "  --a <float>               lattice spacing\n"
        << "  --center_x <float>        default (L-1)/2\n"
        << "  --center_y <float>        default (L-1)/2\n"
        << "  --strength <float>        quartic/Gaussian/double-well/checkerboard/impurity strength\n"
        << "  --sigma <float>           Gaussian width\n"
        << "  --separation <float>      half-separation of double-well minima\n"
        << "  --field <float>           linear potential slope\n"
        << "  --disorder_width <float>  uniform random range width\n"
        << "  --impurity_x <int> --impurity_y <int>\n"
        << "  --seed <int>              disorder seed\n\n"
        << "Outputs:\n"
        << "  <prefix>.svg              gridded potential and occupation heatmaps\n"
        << "  <prefix>.csv              site-by-site numeric values\n";
}

int main(int argc, char** argv) {
    try {
        std::vector<std::string> args;
        for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);

        if (hasFlag(args, "--help")) {
            printHelp();
            return 0;
        }

        Parameters params;
        params.latticeSize = std::stoi(getArgument(args, "--L", std::to_string(params.latticeSize)));
        params.particleCount = std::stoi(getArgument(args, "--N", std::to_string(params.particleCount)));
        params.hoppingT = std::stod(getArgument(args, "--t", std::to_string(params.hoppingT)));
        params.outputPrefix = getArgument(args, "--output", params.outputPrefix);

        const std::string boundary = getArgument(args, "--boundary", "periodic");
        if (boundary == "periodic") params.boundary = BoundaryCondition::Periodic;
        else if (boundary == "open") params.boundary = BoundaryCondition::Open;
        else throw std::runtime_error("--boundary must be periodic or open.");

        const std::string model = getArgument(args, "--model", "boson");
        if (model == "boson" || model == "hardcore_boson") params.model = ParticleModel::HardcoreBoson;
        else if (model == "fermion" || model == "spinless_fermion") params.model = ParticleModel::SpinlessFermion;
        else throw std::runtime_error("--model must be boson or fermion.");

        params.potential = parsePotential(getArgument(args, "--potential", "sho"));
        params.mass = std::stod(getArgument(args, "--mass", std::to_string(params.mass)));
        params.omega = std::stod(getArgument(args, "--omega", std::to_string(params.omega)));
        params.latticeSpacing = std::stod(getArgument(args, "--a", std::to_string(params.latticeSpacing)));
        if (getArgument(args, "--center_x", "unset") != "unset") {
            params.centerX = std::stod(getArgument(args, "--center_x", "0"));
        }
        if (getArgument(args, "--center_y", "unset") != "unset") {
            params.centerY = std::stod(getArgument(args, "--center_y", "0"));
        }
        params.strength = std::stod(getArgument(args, "--strength", std::to_string(params.strength)));
        params.sigma = std::stod(getArgument(args, "--sigma", std::to_string(params.sigma)));
        params.separation = std::stod(getArgument(args, "--separation", std::to_string(params.separation)));
        params.field = std::stod(getArgument(args, "--field", std::to_string(params.field)));
        params.disorderWidth = std::stod(getArgument(args, "--disorder_width", std::to_string(params.disorderWidth)));
        params.impurityX = std::stoi(getArgument(args, "--impurity_x", std::to_string(params.impurityX)));
        params.impurityY = std::stoi(getArgument(args, "--impurity_y", std::to_string(params.impurityY)));
        params.seed = static_cast<unsigned int>(std::stoul(getArgument(args, "--seed", std::to_string(params.seed))));
        params.jacobiSweeps = std::stoi(getArgument(args, "--jacobi_sweeps", std::to_string(params.jacobiSweeps)));
        params.jacobiTolerance = std::stod(getArgument(args, "--jacobi_tol", "1e-12"));

        if (params.latticeSize <= 0) throw std::runtime_error("--L must be positive.");
        if (params.hoppingT <= 0.0) throw std::runtime_error("--t must be positive.");
        if (params.sigma <= 0.0) throw std::runtime_error("--sigma must be positive.");

        const int siteCount = params.latticeSize * params.latticeSize;
        const std::vector<Bond> bonds = makeSquareLatticeBonds(params.latticeSize, params.boundary);
        const std::vector<Mask> basis = makeFixedParticleBasis(siteCount, params.particleCount);
        const std::vector<double> onsitePotential = makeOnsitePotential(params);

        std::cout << "Constructing " << basis.size() << " x " << basis.size()
                  << " Hamiltonian...\n";
        Matrix hamiltonian = makeHamiltonian(params, basis, bonds, onsitePotential);
        EigenSystem eigensystem = jacobiDiagonalize(
            std::move(hamiltonian),
            params.jacobiSweeps,
            params.jacobiTolerance
        );

        double groundEnergy = 0.0;
        const std::vector<double> groundState = groundStateVector(eigensystem, groundEnergy);
        const std::vector<double> occupations = calculateOccupations(basis, groundState, siteCount);

        const double occupationSum = std::accumulate(occupations.begin(), occupations.end(), 0.0);
        const double bondEnergy = calculateBondExpectation(params, basis, bonds, groundState);
        const double potentialEnergy = calculatePotentialExpectation(occupations, onsitePotential);
        const double hoppingEnergy = groundEnergy - bondEnergy - potentialEnergy;

        std::cout << std::fixed << std::setprecision(10);
        std::cout << "Model: " << modelName(params.model) << '\n';
        std::cout << "Boundary: " << boundaryName(params.boundary) << '\n';
        std::cout << "Potential: " << potentialName(params.potential) << '\n';
        std::cout << "Sites: " << siteCount << ", particles: " << params.particleCount
                  << ", basis dimension: " << basis.size() << '\n';
        std::cout << "Bonds: " << bonds.size() << '\n';
        std::cout << "Ground-state energy: " << groundEnergy << '\n';
        std::cout << "  original diagonal bond expectation: " << bondEnergy << '\n';
        std::cout << "  hopping expectation:                " << hoppingEnergy << '\n';
        std::cout << "  onsite-potential expectation:       " << potentialEnergy << '\n';
        std::cout << "Occupation sum: " << occupationSum
                  << " (must equal N=" << params.particleCount << ")\n\n";

        printGrid(onsitePotential, params.latticeSize, 6, "Onsite potential grid:");
        std::cout << '\n';
        printGrid(occupations, params.latticeSize, 6, "Ground-state occupation grid:");

        const std::string csvFile = params.outputPrefix + ".csv";
        const std::string svgFile = params.outputPrefix + ".svg";
        writeCsv(csvFile, params, onsitePotential, occupations);
        writeSvgHeatmap(svgFile, params, onsitePotential, occupations, groundEnergy);

        std::cout << "\nWrote " << csvFile << "\n";
        std::cout << "Wrote " << svgFile << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
