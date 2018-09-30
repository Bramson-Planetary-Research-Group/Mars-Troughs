#include <stdio.h>
#include <stdlib.h>

#include "gsl/gsl_spline.h"
#include "gsl/gsl_spline2d.h"

#include "driver.h"
#include "rk4.h"

void driver(double xi, double zi, double*params, int Np, double*times_out, int N, double*times, int Nt, double*x, double*z, double*insolation, double*lags, int Nl, double*retreats, int do_cleanup){
  
  //Declare some iteration variables
  int i,j;

  //Use this for degbugging
  for (i=0; i < Nl; i++){
    //printf("here:%d %.0f   %.0f\n",i, times[i], lags[i]);
  }

  //CREATE THE INSOLATION AND LAG SPLINES HERE
  static int init_flag = 1;
  static gsl_spline*ins_spl = NULL;
  static gsl_interp_accel*acc = NULL;
  static gsl_interp_accel*xacc = NULL;
  static gsl_interp_accel*yacc = NULL;
  static gsl_spline2d*ret_spl = NULL;

  //FREE THE INSOLATION AND LAG SPLINES HERE
  if (do_cleanup == 1){
    gsl_spline_free(ins_spl);
    gsl_interp_accel_free(acc);
    gsl_spline2d_free(ret_spl);
    gsl_interp_accel_free(xacc);
    gsl_interp_accel_free(yacc);
    init_flag = 1;
    return;
  }

  if (init_flag == 1){
    ins_spl = gsl_spline_alloc(gsl_interp_cspline, Nt);
    acc = gsl_interp_accel_alloc();
    xacc = gsl_interp_accel_alloc();
    yacc = gsl_interp_accel_alloc();
    ret_spl = gsl_spline2d_alloc(gsl_interp2d_bilinear, Nt, Nl);
    
    gsl_spline_init(ins_spl, times, insolation, Nt);
    //Test call
    gsl_spline_eval(ins_spl, 0, acc);
    gsl_spline2d_init(ret_spl, times, lags, retreats, Nt, Nl);
    //Test call
    gsl_spline2d_eval(ret_spl, 0, 13, xacc, yacc);
    init_flag = 0;
  }
  
  //Calculate the time step
  double dt = times_out[1]-times_out[0];

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
  for (i = 0; i < N-1; i++){
    rk4(current_position, dt, t, params, ins_spl, acc, ret_spl, xacc, yacc);
    t+=dt;
      
    x[i+1] = current_position[0];
    z[i+1] = current_position[1];
  }

  //Free the positions array
  free(current_position);

  //Now return the all_positions array
  return;
}
