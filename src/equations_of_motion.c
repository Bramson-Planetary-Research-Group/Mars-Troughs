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
  double alpha = params[0];
  double beta = params[1];
  double gamma = params[2];
  //l0 = params[****]; //Constant lag model
  //Quadratic lag model
  double l0 = params[3];
  double l1 = params[4];
  double l2 = params[5];

  //Calculate accumulation(t) model
  double ins_t = gsl_spline_eval(ins_spl, t, acc)
  double At = alpha*ins_t*ins_t + beta*ins_t + gamma;
  //Calculate the lag(t) model
  //double lag = l0; //Constant lag model
  double lag = l2*t*t + l1*t + l0;
  //
  /* MODELING CHOICES ABOVE */

  //Calculate the retreat
  if ((lag < 0)||(lag > 20)){
      printf("LAGERR: %.1f %.1f\n",t,lag);
      fflush(stdout);
  }
  if ((t < 0)||(t > 1500000.0)){
    printf("TIMEERR: %.1f %.1f\n",t,lag);
    fflush(stdout);
  }
  double Rt = gsl_spline2d_eval(ret_spl, t, lag, xacc, yacc);

  //Calculate the derivatives and return
  k[0] = (Rt + At*cos_t)/sin_t;
  k[1] = -At;

  return;
}
