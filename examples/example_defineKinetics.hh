/* defineKinetics.hh  --  DEMO, written by make_demo_complab.py
 *
 * One microbial reaction:
 *
 *     Ac + A -> P     R = Vmax * Bio * Ac/(Ks_ac+Ac) * A/(Ks_a+A)
 *
 * Chemical order, and it must match CompLaB.xml:
 *     C[0] = Ac    C[1] = A    C[2] = P
 * Microbe order:
 *     B[0] = Bio
 *
 * The constants below are what the collector reads into inputs/kinetics. They
 * are the constants a run of this demo would have been built with.
 */
#ifndef DEFINE_KINETICS_HH
#define DEFINE_KINETICS_HH

#include <vector>
#include <string>
#include <cstddef>
#include <algorithm>

namespace KineticParams {
    constexpr int AC = 0, A = 1, P = 2;          // chemicals
    constexpr int BIO = 0;                       // microbes

    const double Vmax        = 5.0e-5;   // mol/gDW/s   maximum uptake flux
    const double Ks_ac       = 1.0e-4;   // mol/L       half saturation, donor
    const double Ks_a        = 5.0e-5;   // mol/L       half saturation, acceptor
    const double Y           = 0.35;     // gDW/mol     yield on the donor
    const double k_decay     = 1.0e-7;   // 1/s         first order decay
    const double dt_kinetics = 1.25e-2;  // s           one solver step
    const double MAX_RATE_FRACTION = 0.5;// -           per step stability limit
}

namespace KineticsStats {
    static const double MIN_BIOMASS = 1e-12;
    static double iter_sum_dB = 0.0, iter_max_biomass = 0.0, iter_max_dB = 0.0,
                  iter_min_DOC = 1e30;
    static long iter_cells_with_biomass = 0, iter_cells_with_growth = 0;
    inline void resetIteration() {
        iter_sum_dB = 0; iter_max_biomass = 0; iter_max_dB = 0; iter_min_DOC = 1e30;
        iter_cells_with_biomass = 0; iter_cells_with_growth = 0;
    }
    inline void accumulate(double biomass, double donor, double dB) {
        if (biomass > MIN_BIOMASS) {
            iter_cells_with_biomass++; iter_sum_dB += dB;
            if (biomass > iter_max_biomass) iter_max_biomass = biomass;
            if (dB > iter_max_dB) iter_max_dB = dB;
            if (donor < iter_min_DOC && donor > 0) iter_min_DOC = donor;
            if (dB > 0) iter_cells_with_growth++;
        }
    }
    inline void getStats(long& cb, long& cg, double& s, double& mB,
                         double& mdB, double& mD) {
        cb = iter_cells_with_biomass; cg = iter_cells_with_growth;
        s = iter_sum_dB; mB = iter_max_biomass; mdB = iter_max_dB;
        mD = (iter_min_DOC < 1e20) ? iter_min_DOC : 0.0;
    }
}

/* Per reaction rate output, the opt in read by complab3d_rates.hh. */
#define COMPLAB_HAS_RXN_RATES 1
namespace RxnRates {
    inline const std::vector<std::string>& names() {
        static const std::vector<std::string> n = { "Bio_growth_on_Ac_and_A" };
        return n;
    }
}

void defineRxnKinetics(std::vector<double> B, std::vector<double> C,
                       std::vector<double>& subsR, std::vector<double>& bioR,
                       plb::plint mask, std::vector<double>* rxnR = 0)
{
    using namespace KineticParams;
    for (std::size_t i = 0; i < subsR.size(); ++i) subsR[i] = 0.0;
    for (std::size_t i = 0; i < bioR.size();  ++i) bioR[i]  = 0.0;
    if (rxnR) for (std::size_t i = 0; i < rxnR->size(); ++i) (*rxnR)[i] = 0.0;

    if (B.size() < 1 || C.size() < 3 || subsR.size() < 3) return;
    if (mask < 2) return;

    const double bio = std::max(B[BIO], 0.0);
    const double ac  = std::max(C[AC],  0.0);
    const double a   = std::max(C[A],   0.0);

    double R = Vmax * bio * (ac / (Ks_ac + ac)) * (a / (Ks_a + a));

    const double cap = std::min(ac, a) * MAX_RATE_FRACTION / dt_kinetics;
    if (R > cap) R = cap;

    subsR[AC] = -R;
    subsR[A]  = -R;
    subsR[P]  = +R;
    bioR[BIO] = Y * R - k_decay * bio;

    if (!rxnR) KineticsStats::accumulate(bio, ac, bioR[BIO]);
    if (rxnR && rxnR->size() >= 1) (*rxnR)[0] = R;
}

#endif
