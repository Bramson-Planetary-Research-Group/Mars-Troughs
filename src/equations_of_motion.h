#include "gsl/gsl_spline.h"
#include "gsl/gsl_spline2d.h"

void equations_of_motion(double*positions, double*k, double t, double*params, gsl_spline*ins_spl,gsl_interp_accel*acc,gsl_spline2d*ret_spl, gsl_interp_accel*xacc, gsl_interp_accel*yacc);
