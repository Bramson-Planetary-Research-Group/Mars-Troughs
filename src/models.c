//Header files in this project
#include "models.h"

//Built-in libraries
#include <stdio.h>

//GSL Libraries
#include "gsl/gsl_spline.h"
#include "gsl/gsl_spline2d.h"

//Initializer variable
static int init_flag   = 1;
//Insolation splines
static gsl_spline*ins_spl;
static gsl_interp_accel*acc;
//Retreat splines
static gsl_spline2d*ret_spl;
static gsl_interp_accel*xacc;
static gsl_interp_accel*yacc;

void initialize_basic_splines(double*times, int Ntimes, double *insolation, double*lags, int Nlags, double*retreats, int reinit){
  //If the splines haven't been initialized, then do that
  if (init_flag == 1){
    //Allocate
    ins_spl = gsl_spline_alloc(gsl_interp_cspline, Ntimes);
    acc = gsl_interp_accel_alloc();
    xacc = gsl_interp_accel_alloc();
    yacc = gsl_interp_accel_alloc();
    ret_spl = gsl_spline2d_alloc(gsl_interp2d_bilinear, Ntimes, Nlags);
  }
  if (init_flag == 1 || reinit == 1){
    //Initialize
    gsl_spline_init(ins_spl, times, insolation, Ntimes);
    gsl_spline2d_init(ret_spl, times, lags, retreats, Ntimes, Nlags);
    init_flag = 0;
  }
  return;
}

void get_insolation(double*times, int Ntimes, double*insolation){
  int i;
  for(i = 0; i < Ntimes; i++){
    insolation[i] = gsl_spline_eval(ins_spl, times[i], acc);
  }
  return;
}

/* Get the insolation as a function of time given the parameters and the model
 * being specified.
 */
void get_accumulation(int model_flag, double*params, double*times, double*accumulation, int Ntimes){
  int i;
  double ins_t;
  switch(model_flag){
  case 0:
    //A model linear in the insolation.
    for(i = 0; i < Ntimes; i++){
      ins_t = gsl_spline_eval(ins_spl, times[i], acc);
      accumulation[i] = params[0]*ins_t + params[0];
    }
    break;
  default:
    printf("Model not currently implemented.");
  }
  return;
}

void get_lag(){return;}
