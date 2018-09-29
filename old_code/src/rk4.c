#include <math.h>
#include <stdio.h>

#include "gsl/gsl_spline.h"

#include "equations_of_motion.h"

void rk4(double*positions,double dt,double t,double*params,gsl_spline*ins_spl,gsl_interp_accel*acc,gsl_spline2d*ret_spl, gsl_interp_accel*xacc, gsl_interp_accel*yacc){
  /*Fourth order Runge Kutta ODE integrator.
    
    Args:
        positions: current value of all coordinates and velocities
	dt: timestep, seconds
	t: current time, seconds
	params: the free parameters
   */
  //Declare the K arrays that hold the derivatives
  double k1[2];
  double k2[2];
  double k3[2];
  double k4[2];

  //A temporary array
  double temp[2];

  //Iteration variables
  int i,j;

  //Find the derivatives at the start time for k1
  equations_of_motion(positions,k1,t,params,ins_spl,acc,ret_spl,xacc,yacc);

  //Calculate the temp array
  for(i=0;i<2;i++){
    temp[i] = positions[i]+dt/2.*k1[i];
  }

  //Find the derivatives at t+dt/2 for k2
  equations_of_motion(temp,k2,t+dt/2.,params,ins_spl,acc,ret_spl,xacc,yacc);

  //Calculate the new temp arrays
  for(i=0;i<2;i++)
    temp[i] = positions[i]+dt/2.*k2[i];

  //Find the derivatives at t+dt/2 for k3
  equations_of_motion(temp,k3,t+dt/2.,params,ins_spl,acc,ret_spl,xacc,yacc);

  //Calculate the new temp arrays
  for(i=0;i<2;i++)
    temp[i] = positions[i]+dt*k3[i];

  //Find the derivatives at t+dt for k4
  equations_of_motion(temp,k4,t+dt,params,ins_spl,acc,ret_spl,xacc,yacc);

  //Update the positions
  for (i=0;i<2;i++){
    positions[i] = positions[i] + dt/6.*(k1[i]+2.*k2[i]+2.*k3[i]+k4[i]);
  }
  //End rk4
  return;
}
