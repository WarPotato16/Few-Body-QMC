#ifndef HARMONIC_POTENTIAL_HPP
#define HARMONIC_POTENTIAL_HPP

#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace sho {

struct Parameters {
    bool enabled = false;
    double mass = 1.0;
    double omega = 1.0;
    double latticeSpacing = 1.0;
    double centerX = std::numeric_limits<double>::quiet_NaN();
    double centerY = std::numeric_limits<double>::quiet_NaN();

    double shiftEpsilon = 1e-12;
};

struct HarmonicPotential {
    bool enabled = false;
    int latticeSize = 0;
    double centerX = 0.0;
    double centerY = 0.0;
    std::vector<double> sitePotential;
    std::vector<double> siteShift;
    double totalShift = 0.0;

    double valueAt(int site) const {
        return enabled ? sitePotential.at(site) : 0.0;
    }

    double shiftedMatrixElement(int site, bool occupied) const {
        if (!enabled) {
            return 0.0;
        }
        return siteShift.at(site) - (occupied ? sitePotential.at(site) : 0.0);
    }

    double statePotentialEnergy(unsigned int state) const {
        if (!enabled) {
            return 0.0;
        }

        double energy = 0.0;
        for (int site = 0; site < static_cast<int>(sitePotential.size()); ++site) {
            if (((state >> site) & 1U) != 0U) {
                energy += sitePotential[site];
            }
        }
        return energy;
    }
};

inline HarmonicPotential makeHarmonicPotential(int latticeSize, const Parameters& params) {
    if (latticeSize <= 0) {
        throw std::runtime_error("latticeSize must be positive when constructing the harmonic trap.");
    }

    HarmonicPotential trap;
    trap.enabled = params.enabled;
    trap.latticeSize = latticeSize;

    const double defaultCenter = 0.5 * static_cast<double>(latticeSize - 1);
    trap.centerX = std::isnan(params.centerX) ? defaultCenter : params.centerX;
    trap.centerY = std::isnan(params.centerY) ? defaultCenter : params.centerY;

    const int siteCount = latticeSize * latticeSize;
    trap.sitePotential.assign(siteCount, 0.0);
    trap.siteShift.assign(siteCount, 0.0);

    if (!params.enabled) {
        return trap;
    }
    if (params.mass <= 0.0) {
        throw std::runtime_error("Harmonic-trap mass must be positive.");
    }
    if (params.omega <= 0.0) {
        throw std::runtime_error("Harmonic-trap omega must be positive when --trap is enabled.");
    }
    if (params.latticeSpacing <= 0.0) {
        throw std::runtime_error("Harmonic-trap lattice spacing must be positive.");
    }
    if (params.shiftEpsilon <= 0.0) {
        throw std::runtime_error("Harmonic-trap shift epsilon must be positive.");
    }

    for (int y = 0; y < latticeSize; ++y) {
        for (int x = 0; x < latticeSize; ++x) {
            const int site = y * latticeSize + x;
            const double dx = (static_cast<double>(x) - trap.centerX) * params.latticeSpacing;
            const double dy = (static_cast<double>(y) - trap.centerY) * params.latticeSpacing;
            const double radiusSquared = dx * dx + dy * dy;
            const double potential = 0.5 * params.mass * params.omega * params.omega * radiusSquared;

            trap.sitePotential[site] = potential;
            trap.siteShift[site] = potential + params.shiftEpsilon;
        }
    }

    trap.totalShift = std::accumulate(trap.siteShift.begin(), trap.siteShift.end(), 0.0);
    return trap;
}

}
#endif
