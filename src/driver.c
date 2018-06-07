#include <stdio.h>
#include <stdlib.h>

#include "gsl/gsl_spline.h"
#include "gsl/gsl_spline2d.h"

#include "driver.h"
#include "rk4.h"

void driver(double xi, double zi, double*params, int Np, double*times, int N, double*x, double*z, double*insolation, double*lags, int Nl, double*retreats){  
  
  //Declare some iteration variables
  int i,j;

  //CREATE THE INSOLATION AND LAG SPLINES HERE
  gsl_spline*ins_spl = gsl_spline_alloc(gsl_interp_cspline, N);
  gsl_interp_accel*acc = gsl_interp_accel_alloc();
  gsl_spline_init(ins_spl, times, insolation, N);
  //gsl_spline_eval(ins_spl, t, acc);
  gsl_interp_accel*xacc = gsl_interp_accel_alloc();
  gsl_interp_accel*yacc = gsl_interp_accel_alloc();
  gsl_spline2d*ret_spl = gsl_spline2d_alloc(gsl_interp2d_bilinear, N, Nl);
  gsl_spline2d_init(ret_spl, times, lags, retreats, N, Nl);
  //gsl_spline2d_eval(ret_spl, time, lag, xacc, yacc);

  //Calculate the time step
  double dt = times[1]-times[0];

  //Declare the current time
  double t=0;

  double*current_position = (double*)malloc(2*sizeof(double));

  //Put in the current position as the initial position
  x[0] = xi;
  z[0] = zi;
  current_position[0] = xi;
  current_position[1] = zi;
	
  //Loop over all the times and advance the simulation
  //Note: we take N-2 steps since we have already stored the initial information
  for (i = 0; i < N-2; i++){
    rk4(current_position, dt, t, params, ins_spl, acc, ret_spl, xacc, yacc);
    t+=dt;
      
    x[i+1] = current_position[0];
      z[i+1] = current_position[1];
  }

  //Free the positions array
  free(current_position);

  //FREE THE INSOLATION AND LAG SPLINES HERE
  gsl_spline_free(ins_spl);
  gsl_interp_accel_free(acc);
  gsl_spline2d_free(ret_spl);
  gsl_interp_accel_free(xacc);
  gsl_interp_accel_free(yacc);

  //Now return the all_positions array
  return;
}
