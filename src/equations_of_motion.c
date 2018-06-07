#include <math.h>
#include <stdio.h>

#include "equations_of_motion.h"

//The angle is 2.9 degrees. Pre-evaluate the trig functions
#define sin_t 0.0505929400767133
#define cos_t 0.9987193571841863

void equations_of_motion(double*positions, double*k, double t, double*params, gsl_spline*ins_spl,gsl_interp_accel*acc,gsl_spline2d*ret_spl, gsl_interp_accel*xacc, gsl_interp_accel*yacc){

  //Read out x and z
  double x,z;
  x = positions[0];
  z = positions[1];

  /* MODELING CHOICES BELOW */
  //
  //Read out the free parameters
  double beta, gamma, l0;
  beta = params[0];
  gamma = params[1];
  l0 = params[2]; //Constant lag model
  //Calculate accumulation(t) model
  double At = beta*(gsl_spline_eval(ins_spl, t, acc) - gamma);
  //Calculate the lag(t) model
  double lag = l0; //Constant lag model
  //
  /* MODELING CHOICES ABOVE */

  //Calculate the retreat
  double Rt = gsl_spline2d_eval(ret_spl, t, lag, xacc, yacc);

  //Calculate the derivatives and return
  k[0] = (Rt + At*cos_t)/sin_t;
  k[1] = -At;

  return;
}
