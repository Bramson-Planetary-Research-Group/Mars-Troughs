"""This has the trough object with all the necessary functions.
"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import RectBivariateSpline as RBS
import os, inspect
here = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))+"/"

class Trough(object):
    def __init__(self, acc_params, lag_params, acc_model_number, lag_model_number, errorbar):
        """Constructor for the trough object.

        Args:
            acc_params (array like): model parameters for accumulation
            acc_model_number (int): index of the accumulation model
            lag_params (array like): model parameters for lag(t)
            lag_model_number (int): index of the lag(t) model
            errorbar (float): errorbar of the real data
        """
        #Trough angle
        self.angle_degrees = 2.9 #degrees
        self.sin_angle = np.sin(self.angle_degrees * np.pi/180.)
        self.cos_angle = np.cos(self.angle_degrees * np.pi/180.)
        self.csc_angle = 1./self.sin_angle
        self.sec_angle = 1./self.cos_angle
        self.tan_angle = self.sin_angle/self.cos_angle
        self.cot_angle = 1./self.tan_angle
        #Set up the trough model
        self.acc_params = np.array(acc_params)
        self.lag_params = np.array(lag_params)
        self.acc_model_number = acc_model_number
        self.lag_model_number = lag_model_number
        self.errorbar = errorbar
        #Load in supporting data
        insolation, ins_times = np.loadtxt(here+"/Insolation.txt", skiprows=1).T
        ins_times = -ins_times #positive times are now in the past
        self.insolation = insolation
        self.ins_times  = ins_times
        #Create the look up table for retreats
        self.lags       = lags = np.arange(16)+1
        self.lags[0] -= 1
        self.lags[-1] = 20
        self.retreats   = np.loadtxt(here+"/R_lookuptable.txt").T
        #Splines
        self.ins_spline = IUS(ins_times, insolation)
        self.iins_spline = self.ins_spline.antiderivative()
        self.ins2_spline = IUS(ins_times, insolation**2)
        self.iins2_spline = self.ins2_spline.antiderivative()
        self.ret_spline = RBS(self.lags, self.ins_times, self.retreats)
        self.re2_spline = RBS(self.lags, self.ins_times, self.retreats**2)
        #Pre-calculate the lags at all times
        self.lags_t = self.get_lag_at_t(self.ins_times)
        self.lag_spline = IUS(ins_times, self.lags_t)
        self.Retreat_model_at_t = self.get_Rt_model(self.lags_t, self.ins_times)
        self.retreat_model_spline = IUS(ins_times, self.Retreat_model_at_t)
        self.iretreat_model_spline = self.retreat_model_spline.antiderivative()
        #Load in the real data
        #xdata, ydata = np.loadtxt(here+"/RealXandZ.txt")
        xdata, ydata = np.loadtxt(here+"/TMP_xz.txt", unpack=True)
        self.xdata = xdata*1000 #convert to meters
        self.ydata = ydata #meters
        self.Ndata = len(self.xdata) #number of data points
        
    def set_model(self, acc_params, lag_params):
        """Setup a new model, with new accumulation and lag parameters.
        """
        assert len(acc_params)==len(self.acc_params), \
            "New and original accumulation parameters must have the same shape."
        assert len(lag_params)==len(self.lag_params), \
            "New and original lag parameters must have the same shape."
        self.acc_params = acc_params
        self.lag_params = lag_params
        self.lags_t = self.get_lag_at_t(self.ins_times)
        self.lag_spline = IUS(ins_times, self.lags_t)
        self.Retreat_model_at_t = self.get_Rt_model(self.lags_t, self.ins_times)
        self.retreat_model_spline = IUS(ins_times, self.Retreat_model_at_t)
        self.iretreat_model_spline = self.retreat_model_spline.antiderivative()
        return
        
    def get_insolation(self, time):
        return self.ins_spline(time)
        
    def get_retreat(self, lag, time):
        return self.ret_spline(time, lag)

    def get_lag_at_t(self, time): #Model dependent
        num = self.lag_model_number
        p = self.lag_params
        if num == 0:
            a = p[0] #lag = constant
            return a*np.ones_like(time)
        if num == 1:
            a,b = p[0:2] #lag(t) = a + b*t
            return a + b*time
        return #error, since no number is returned

    def get_Rt_model(self, lags, times):
        ret = self.ret_spline.ev(lags, times)
        return ret

    def get_accumulation(self, time): #Model dependent
        num = self.acc_model_number
        p = self.acc_params
        if num == 0:
            a = p[0] #a*Ins(t)
            return -1*(a*self.ins_spline(time))
        if num == 1:
            a,b = p[0:2] #a*Ins(t) + b*Ins(t)^2
            return -1*(
                a*self.ins_spline(time)+
                b*self.ins2_spline(time))
        return #error, since no number is returned

    def get_yt(self, time): #Model dependent
        #This is the depth the trough has traveled
        num = self.acc_model_number
        p = self.acc_params
        if num == 0:
            a = p[0] #a*Ins(t)
            return -1*(
                a*(self.iins_spline(time) - self.iins_spline(0)))
        if num == 1:
            a,b = p[0:2] #a*Ins(t) + b*Ins(t)^2
            return -1*(
                a*(self.iins_spline(time) - self.iins_spline(0))+
                b*(self.iins2_spline(time) - self.iins2_spline(0)))
        return #error

    def get_xt(self, time):
        #This is the horizontal distance the trough has traveled
        #Model dependent
        yt = self.get_yt(time)
        return -self.cot_angle*yt + self.sec_angle * (self.iretreat_model_spline(time) - self.iretreat_model_spline(0))

    def get_trajectory(self):
        x = self.get_xt(self.ins_times)
        y = self.get_yt(self.ins_times)
        return x,y

    def get_nearest_points(self):
        x = self.get_xt(self.ins_times)
        y = self.get_yt(self.ins_times)
        xd = self.xdata
        yd = self.ydata
        xn = np.zeros_like(xd)
        yn = np.zeros_like(yd)
        for i in range(len(xd)):
            ind = np.argmin((x-xd[i])**2 + (y-yd[i])**2)
            xn[i] = x[ind]
            yn[i] = y[ind]
        return xn, yn

    def lnlikelihood(self):
        xd = self.xdata
        yd = self.ydata
        xn, yn = self.get_nearest_points()
        chi2 = (yd-yn)**2/self.errorbar**2
        return -0.5*chi2.sum() - self.Ndata*np.log(self.errorbar)

if __name__ == "__main__":
    print("Test main() in trough.py")
    test_acc_params = [1e-8, 5e-9]
    acc_model_number = 1
    test_lag_params = [1, 1e-9]
    lag_model_number = 1
    errorbar = 100.
    tr = Trough(test_acc_params, test_lag_params,
                acc_model_number, lag_model_number,
                errorbar)
    
    import matplotlib.pyplot as plt
    #plt.rc("text", usetex=True)
    plt.rc("font", size=14, family='serif')
    times = tr.ins_times
    fig, ax = plt.subplots(ncols=2, nrows=3, sharex=True)
    ax[0,0].set_ylabel("Ins(t)")
    ax[1,0].set_ylabel("-A(t)")
    ax[2,0].set_ylabel(r"$y(t)$")# = \int_0^t{\rm d}t'\ A(t')$")
    ax[2,0].set_xlabel(r"Lookback Time (yrs)")
    ax[2,1].set_xlabel(r"Lookback Time (yrs)")
    ax[1,0].ticklabel_format(style='scientific', scilimits=(0,0))#, useMathText=True)
    ax[2,0].ticklabel_format(axis='x', style='sci', scilimits=(0,0))#, useMathText=True)
    ax[2,1].ticklabel_format(axis='x', style='sci', scilimits=(0,0))#, useMathText=True)
    ax[0,0].plot(times, tr.get_insolation(times))
    ax[1,0].plot(times, tr.get_accumulation(times))
    ax[2,0].plot(times, tr.get_yt(times))

    ax[0,1].plot(times, tr.get_lag_at_t(times))
    ax[1,1].plot(times, tr.retreat_model_spline(times))
    ax[2,1].plot(times, tr.get_xt(times))
    for i in range(3):
        ax[i,1].yaxis.tick_right()
        ax[i,1].yaxis.set_label_position("right")
    ax[0,1].set_ylabel(r"Lag(t) (mm)")
    ax[1,1].set_ylabel(r"$R(t)$ (km)")
    ax[2,1].set_ylabel(r"$x(t)$")# = \int_0^t{\rm d}t' \frac{R(t') + A(t')\cos(\theta)}{\sin(\theta)}$")
    ax[2,1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))#, useMathText=True)

    plt.subplots_adjust(wspace=0.1)
    fig.savefig("example_plot.png", dpi=300, bbox_inches='tight')
    #plt.show()
    plt.clf()

    #Compare the trajectory with data
    plt.plot(tr.get_xt(times), tr.get_yt(times))
    plt.errorbar(tr.xdata, tr.ydata, yerr=tr.errorbar, c='k', marker='.', ls='')
    xn,yn = tr.get_nearest_points()
    plt.plot(xn, yn, ls='', marker='o', c='r')
    print(tr.lnlikelihood())
    plt.show()
