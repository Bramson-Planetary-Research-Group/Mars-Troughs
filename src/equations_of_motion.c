#include <math.h>
#include <stdio.h>

#include "equations_of_motion.h"

//The angle is 2.9 degrees. Pre-evaluate the trig functions
#define sin_t 0.0505929400767133
#define cos_t 0.9987193571841863

void equations_of_motion(double*positions, double*k, double t, double*params, gsl_spline*ins_spl,gsl_interp_accel*acc,gsl_spline2d*ret_spl, gsl_interp_accel*xacc, gsl_interp_accel*yacc){

  //Check if the time is too high
  if ((t < 0)||(t > 1500000.0)){
    printf("TIMEERR: %.1f\n",t);
    fflush(stdout);
    k[0] = 0;
    k[1] = 0;
    return;
  }

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
  double a = params[3];
  double b = params[4];
  double c = 5.0;//params[5];

  //printf("%e %e %e %e %e %e\n",alpha, beta, gamma, a, b, c);
  //Calculate accumulation(t) model
  double ins_t = gsl_spline_eval(ins_spl, t, acc);
  double At = alpha*ins_t*ins_t + beta*ins_t + gamma;
  //Calculate the lag(t) model
  //double lag = l0; //Constant lag model
  double lag = a*t*t + b*t + c;
  //
  /* MODELING CHOICES ABOVE */

  //Check if the lag is too large or has gone negative
  if ((lag < 0)||(lag > 20.0)){
    //printf("LAGERR: %.1f %.1e\n",t,lag);
    //fflush(stdout);
    k[0] = 0;
    k[1] = 0;
    return;
  }
  //Calculate the retreat
  double Rt = gsl_spline2d_eval(ret_spl, t, lag, xacc, yacc);

  //Calculate the derivatives and return
  k[0] = (Rt + At*cos_t)/sin_t;
  k[1] = -At;

  return;
}
