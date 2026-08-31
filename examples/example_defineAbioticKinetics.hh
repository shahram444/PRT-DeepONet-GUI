/* defineAbioticKinetics.hh  --  DEMO, written by make_demo_complab.py
 *
 * This demo has no chemical reaction without a microbe, so every rate here is
 * zero and <enable_abiotic_kinetics> is false in CompLaB.xml. The file exists
 * because the solver includes it unconditionally, and because the collector
 * records that it was found rather than leaving a hole in inputs/files.
 */
#ifndef DEFINE_ABIOTIC_KINETICS_HH
#define DEFINE_ABIOTIC_KINETICS_HH

#include <vector>
#include <cstddef>

void defineAbioticRxnKinetics(std::vector<double> C, std::vector<double>& subsR,
                              plb::plint mask, std::vector<double>* rxnR = 0)
{
    (void) C; (void) mask;
    for (std::size_t i = 0; i < subsR.size(); ++i) subsR[i] = 0.0;
    if (rxnR) for (std::size_t i = 0; i < rxnR->size(); ++i) (*rxnR)[i] = 0.0;
}

#endif
