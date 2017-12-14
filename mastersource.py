#!/usr/bin/env python


import os
import scipy
import Utilities
import pylab
import copy
import scipy.interpolate
import numpy
import scipy.optimize


class PureTemperatureProperty:

    '''
    The PureTemperatureProperty class of objects represent interpolated function of temperature.
    
    They have the following attributes:
    
    Evaluate                       : This is a method obtained by interpolation that takes temperature in K to evaluate the pure temperature property at that T.
    
    Tmin                           : Is a scalar of the minimum temperature for which the evaluation is valid (K)
    
    Tmax                           : Is a scalar for the maximum temperature for which the evaluation is valid (K)    
    '''
    def __init__(self):
        self.Tmin                  = 'junk'
        self.Tmax                  = 'junk'
        self.Evaluate              = 'junk'

class Molecule:
    '''
    The Molecule class of objects represent pure component IDEAL GAS properties of molecules.
    
    ***************************************************************************
   
   The following are the SCALAR attributes of this class:
   
   ***************************************************************************
    Name                            : The name of the molecule.  The data for the molecule is stored in a text file [Name].txt
    MolecularWeight                 : Is the molecular weight in kg/kmol
    NormalBoilingPoint              : The molecule's normal boiling point in K
    FreezingPoint                   : The molecule's normal freezing point in K
    CriticalTemperature             : The molecule's critical temperature in K
    CriticalPressure                : The molecule's critical pressure in Pa
    CriticalVolume                  : The molecule's critical molar volume in m3/kmol
    AcentricFactor                  : The molecule's acentric factor
    ReferenceTemperature            : The reference temperature at which enthalpy of formation is calculated (298.15 K)
    ReferencePressure               : The reference pressure at which entropy is calculated (101,325 Pa)
    EnthalpyFormation               : The enthalpy of formation of the molecule from its constitutent elements in the ideal gas state at 298.15K.  (J/kmol)
    GibbsEnergyFormation            : The Gibbs free energy of formation of the molecule in the ideal gas state at 298.15 K. (J/kmol)
    EntropyReference                : The entropy of the molecule as an ideal gas at 298.15K and 101,325 Pa pressure. (J/kmol-K)
    ************************************************************************************************************************
    The following private attributes are lists of parameters used in correlating temperature dependent pure component properties.
    ************************************************************************************************************************
    __listVapourPressureParameters          : List of parameters [c1, c2, c3, c4, c5, Tmin, Tmax] used in vapour pressure correlation       
                                            :: vp (Pa)          = exp(c1 + c2/T + c3*log(T) + c4*T**c5).  
                                            The correlation is valid between Tmin and Tmax. T in K
    __listLiquidDensityParameters           : List of parameters [c1, c2, c3, c4, Tmin, Tmax] used in liquid density correlation            
                                            :: dens (kmol/m3)   = c1/c2**(1+(1-T/c3)**c4). 
                                            The correlation is valid between Tmin and Tmax. T in K
    __listEnthalpyVapourizationParameters   : List of parameters [c1, c2, c3, c4, Tmin, Tmax] used in enthalpy of vapourization correlation 
                                            :: hvap (J/kmol)    = c1*(1-Tr)**(c2+c3*Tr+c4*Tr*Tr). 
                                            Where Tr = T/Tc. Tc = Critical Temperature in K, T in K.  
                                            Tmin and Tmax are the boundaries of the valid region for this correlation.
    __listSpecificHeatCapacityParameters    : List of parameters [c1, c2, c3, c4, c5, Tmin, Tmax] used in IDEAL GAS specific heat capacity  
                                            :: cp (J/kmol-K)    = c1 + c2*(csin/scipy.sinh(csin))**2+c4*(ccos/scipy.cosh(ccos))**2. 
                                            Here, csin = c3/T and ccos = c4/T and T is in K.  
                                            Tmin and Tmax are the boundaries of the valid region for the correlation.
    **************************************************************************************************************************
    The object contains the following private utility functions
    **************************************************************************************************************************
    __CorrelationBank                         : Calculates the given property at the given temperature using correlations.
    __readCorrelation                         : A method to read parameters for and verify given correlations against endpoint values
    __readScalar                              : A method to read and verify scalar properties
    __readDataFile                            : A method to read the data file [name].txt
    **************************************************************************************************************************
    The following pure temperature property methods are available.
    EXAMPLE:  Take the method VapourPressure,
                VapourPressure('bounds') returns a list [Tmin, Tmax] specifying the bounds of the temperature range where it is valid.
                VapourPressure(T) returns a float which is the vapour pressure at temperature T (in K)
    The methods use a UnivariateSpline object to interpolate properties are functions of temperature.
    **************************************************************************************************************************
    VapourPressure(T)                          : Vapour Pressure in Pa as a function of T.  
                                                Interpolation from 1000 points between Tmin and Tmax. Cubic spline.
    LiquidDensity(T)                           : Liquid Density in kmol/m3 as a function of T.
                                                Interpolation from 1000 points between Tmin and Tmax. Cubic spline.
    EnthalpyVapourization(T)                   : Enthalpy of Vapourization in J/kmol as a function of T.
                                                Interpolation from 1000 points between Tmin and Tmax. Cubic spline.
    SpecificHeatCapacity(T)                    : Specific Heat Capacity of Ideal Gas J/kmol-K as a function of T.
                                                Interpolation from 1000 points between Tmin and Tmax. Cubic spline.
    EnthalpyIdealGas(T)                        : Enthalpy of Ideal Gas J/kmol as a function of T.
                                                Interpolation from 1000 points between Tmin and Tmax. Cubic spline.
                                                H = Hfig + [Integral SpecificHeatCapacity from Tref to T]
    EntropyIdealGasReference(T)                : Entropy of Ideal Gas J/kmol-K as a function of T at reference pressure 101,325 Pa.
                                                Interpolation from 1000 points between Tmin and Tmax. Cubic spline.
                                                Sig_ref = Sreference + [Integral SpecificHeatCapacity/T from Tref to T]
                                                Note:  The reference conditions for entropy are Treference = 298.15K and Preference = 101,325 Pa
    GibbsEnergyIdealGasReference(T)            : Gibbs Energy of Ideal Gas J/kmol as a function of T at pressure of 101,325 Pa.
                                                Interpolation from 1000 points between Tmin and Tmax. Cubic spline.
                                                Note: Gfig != Hfig - Tref*Sreference,
                                                    Hence: Gig_ref = Gfig + (H - Hfig) - T*(S - Sreference) 
    '''
    def __init__(self, name):
        '''
        The class intanciation.
       
        obj = Molecule(name) requires the name of the molecule to be specified.
        
        This will direct the code to read the datafile called [name].txt and use to data to ready the object for use.
        
        '''
        self.Name                       =  name                         
        self.MolecularWeight            = 'junk'               
        self.NormalBoilingPoint         = 'junk'  
        self.FreezingPoint              = 'junk'                
        self.CriticalTemperature        = 'junk'               
        self.CriticalPressure           = 'junk'                
        self.CriticalVolume             = 'junk'                
        self.AcentricFactor             = 'junk'             
        self.ReferenceTemperature       = 'junk'
        self.ReferencePressure          = 'junk'             
        self.EnthalpyFormation          = 'junk'        
        self.GibbsEnergyFormation       = 'junk'
        self.EntropyReference           = 'junk'              
        self.__listVapourPressureParameters           = 'junk'      
        self.__listLiquidDensityParameters            = 'junk' 
        self.__listEnthalpyVapourizationParameters    = 'junk'    
        self.__listSpecificHeatCapacityParameters     = 'junk'  
        self.__VapourPressure             = 'junk'
        self.__LiquidDensity              = 'junk' 
        self.__EnthalpyVapourization      = 'junk'
        self.__SpecificHeatCapacity       = 'junk'
        self.__EnthalpyIdealGas           = 'junk'
        self.__EntropyIdealGas            = 'junk'
        
        self.listparams = []   
        self.__readDataFile()
        
        '''
        VapourPressure
        '''
        property = 'VapourPressure'
        Tmin = self.__listVapourPressureParameters[-2]
        Tmax = self.__listVapourPressureParameters[-1]
        xx = scipy.linspace(Tmin, Tmax, 1000)
        yy = self.__CorrelationBank(property, xx)        
        self.__VapourPressure = PureTemperatureProperty()
        self.__VapourPressure.Tmin = Tmin
        self.__VapourPressure.Tmax = Tmax
        self.__VapourPressure.Evaluate = scipy.interpolate.UnivariateSpline(xx, yy)
        
        '''
        Liquid Density
        '''
        property = 'LiquidDensity'
        Tmin = self.__listLiquidDensityParameters[-2]
        Tmax = self.__listLiquidDensityParameters[-1]
        xx = scipy.linspace(Tmin, Tmax, 1000)
        yy = self.__CorrelationBank(property, xx)        
        self.__LiquidDensity = PureTemperatureProperty()
        self.__LiquidDensity.Tmin = Tmin
        self.__LiquidDensity.Tmax = Tmax
        self.__LiquidDensity.Evaluate = scipy.interpolate.UnivariateSpline(xx, yy)
        
        '''
        Enthalpy of Vapourization
        '''
        property = 'EnthalpyVapourization'
        Tmin = self.__listEnthalpyVapourizationParameters[-2]
        Tmax = self.__listEnthalpyVapourizationParameters[-1]
        xx = scipy.linspace(Tmin, Tmax, 1000)
        yy = self.__CorrelationBank(property, xx)        
        self.__EnthalpyVapourization = PureTemperatureProperty()
        self.__EnthalpyVapourization.Tmin = Tmin
        self.__EnthalpyVapourization.Tmax = Tmax
        self.__EnthalpyVapourization.Evaluate = scipy.interpolate.UnivariateSpline(xx, yy)
        
        '''
        Specific Heat Capacity
        '''
        property = 'SpecificHeatCapacity'
        Tmin = self.__listSpecificHeatCapacityParameters[-2]
        Tmax = self.__listSpecificHeatCapacityParameters[-1]
        xx = scipy.linspace(Tmin, Tmax, 1000)
        yy = self.__CorrelationBank(property, xx)        
        self.__SpecificHeatCapacity = PureTemperatureProperty()
        self.__SpecificHeatCapacity.Tmin = Tmin
        self.__SpecificHeatCapacity.Tmax = Tmax
        self.__SpecificHeatCapacity.Evaluate = scipy.interpolate.UnivariateSpline(xx, yy)
        
        '''
        EnthalpyIdealGas:: Hig(T) = Hfig + integral(Cp, Tref, T)
        '''
        Tmin = self.__listSpecificHeatCapacityParameters[-2]
        Tmax = self.__listSpecificHeatCapacityParameters[-1]
        xx = scipy.linspace(Tmin, Tmax, 1000)
        listtemp = []
        for T in xx:
            h =  self.__SpecificHeatCapacity.Evaluate.integral(self.ReferenceTemperature, T)
            listtemp.append(h)
        self.__EnthalpyIdealGas = PureTemperatureProperty()
        self.__EnthalpyIdealGas.Tmin = Tmin
        self.__EnthalpyIdealGas.Tmax = Tmax
        self.__EnthalpyIdealGas.Evaluate = scipy.interpolate.UnivariateSpline(xx, scipy.array(listtemp))
        
        '''
        EntropyIdealGasReference:: Sig(T) = integral(Cp/T, Tref, T)
        '''
        Tmin = self.__listSpecificHeatCapacityParameters[-2]
        Tmax = self.__listSpecificHeatCapacityParameters[-1]
        xx = scipy.linspace(Tmin, Tmax, 1000)
        listtemp = []
        for T in xx:
            cpbyT = self.__CorrelationBank('SpecificHeatCapacity',T)/T
            listtemp.append(cpbyT)
        cpbyT = scipy.interpolate.UnivariateSpline(xx, scipy.array(listtemp))
        listtemp = []
        for T in xx:
            s = cpbyT.integral(self.ReferenceTemperature, T)
            listtemp.append(s)
        self.__EntropyIdealGasReference = PureTemperatureProperty()
        self.__EntropyIdealGasReference.Tmin = Tmin
        self.__EntropyIdealGasReference.Tmax = Tmax
        self.__EntropyIdealGasReference.Evaluate = scipy.interpolate.UnivariateSpline(xx, scipy.array(listtemp))
        
        '''
        GibbsEnergyIdealGasReference:: Gig(T) = Gfig + Hig(T) - Hfig - T*(Sig(T) - Sreference)
        '''
        Tmin = self.__listSpecificHeatCapacityParameters[-2]
        Tmax = self.__listSpecificHeatCapacityParameters[-1]
        xx = scipy.linspace(Tmin, Tmax, 1000)
        yy =  self.GibbsEnergyFormation + self.__EnthalpyIdealGas.Evaluate(xx) - self.EnthalpyFormation + xx*(self.__EntropyIdealGasReference.Evaluate(xx) - self.EntropyReference)
        self.__GibbsEnergyIdealGasReference = PureTemperatureProperty()
        self.__GibbsEnergyIdealGasReference.Tmin = Tmin
        self.__GibbsEnergyIdealGasReference.Tmax = Tmax
        self.__GibbsEnergyIdealGasReference.Evaluate = scipy.interpolate.UnivariateSpline(xx, yy)
        
    
    def VapourPressure(self, T, der):
        '''
        
        
            
            If T is 'bounds', returns a list [Tmin, Tmax] which are the temperature bounds.
            If T is a float,
                if der = 0
                    then returns vapour pressure in Pa.
                elif der = 1
                    then returns first derivative of vapour pressure wrt T (Pa/K)
                elif der = 2
                    then returns second derivative of vapour pressure wrt T (Pa/K)
                else returns valueerror    
        '''
        if T == 'bounds':
            return [self.__VapourPressure.Tmin, self.__VapourPressure.Tmax]
        else:
            if der == 0:
                return self.__VapourPressure.Evaluate(T)
            elif der == 1:
                return self.__VapourPressure.derivatives(T)[1]
            elif der == 2:
                return self.__VapourPressure.derivatives(T)[2]
            else:
                raise ValueError('der values must be in [0,1,2]')
    
    def LiquidDensity(self, T, der):
        '''
       
       LiquidDensity(T):
            If T == 'bounds', then returns a list [Tmin, Tmax] which are the bounds of the correlation
            If T is a float, 
                then if der = 0:
                    returns liquid density (kmol/m3) at the value of T in K
                if der = 1:
                    returns first derivative of liquid density wrt. T
                if der = 2:
                    returns the second derivative
                else:
                    raises ValueError
        '''
        if T == 'bounds':
            return [self.__LiquidDensity.Tmin, self.__LiquidDensity.Tmax]
        else:
            if der == 0:
                return self.__LiquidDensity.Evaluate(T)
            elif der == 1:
                return self.__LiquidDensity.derivatives(T)[1]
            elif der == 2:
                return self.__LiquidDensity.derivatives(T)[2]
            else:
                raise ValueError('der values must be in [0,1,2]')
    
    def EnthalpyVapourization(self, T, der):
        '''
        
        
            If T is 'bounds', returns a list [Tmin, Tmax] which are the temperature bounds.
            If T is a float, 
                then if der = 0:
                    returns enthalpy of vapourization in J/kmol
                else if der = 1:
                    returns the first derivative of Hvap wrt T
                else if der = 2:
                    returns the second derivative
                else raised ValueError
        '''
        if T == 'bounds':
            return [self.__EnthalpyVapourization.Tmin, self.__EnthalpyVapourization.Tmax]
        else:
            if der == 0:
                return self.__EnthalpyVapourization.Evaluate(T)
            elif der == 1:
                return self.__EnthalpyVapourization.derivatives(T)[1]
            elif der == 2:
                return self.__EnthalpyVapourization.derivatives(T)[2]
            else:
                raise ValueError('der values must be in [0,1,2]')
    
    def SpecificHeatCapacity(self, T, der):
        '''
        
            If T is 'bounds', returns a list [Tmin, Tmax] which are the temperature bounds.
            If T is a float, 
                if der = 0:
                    then returns specific heat capacity in ideal gas state in J/kmol-K
                if der = 1:
                    returns the first derivative
                if der = 2:
                    returns the second derivative
                else raises ValueError
        '''
        if T == 'bounds':
            return [self.__SpecificHeatCapacity.Tmin, self.__SpecificHeatCapacity.Tmax]
        else:
            if der == 0:
                return self.__SpecificHeatCapacity.Evaluate(T)
            elif der == 1:
                return self.__SpecificHeatCapacity.derivatives(T)[1]
            elif der == 2:
                return self.__SpecificHeatCapacity.derivatives(T)[2]
            else:
                raise ValueError('der values must be in [0,1,2]')
                
    
    def EnthalpyIdealGas(self, T, der):
        '''
        
      
            If T is 'bounds', returns a list [Tmin, Tmax] which are the temperature bounds.
            If T is a float, 
                if der = 0: then returns enthalpy change in ideal gas state from Tref to T in J/kmol
                if der = 1: returns the first derivative wrt T
                if der = 2: returns the second derivative
                else raises ValueError
        EnthalpyIdealGas:: Hig(T) = integral(Cp, Tref, T)
        '''
        if T == 'bounds':
            return [self.__EnthalpyIdealGas.Tmin, self.__EnthalpyIdealGas.Tmax]
        else:
            if der == 0:
                return self.__EnthalpyIdealGas.Evaluate(T)
            elif der == 1:
                return self.__EnthalpyIdealGas.Evaluate.derivatives(T)[1]
            elif der == 2:
                return self.__EnthalpyIdealGas.Evaluate.derivatives(T)[2]
            else:
                raise ValueError('der values must be in [0,1,2]')
                
    def EntropyIdealGasReference(self, T, der):
        '''
        
            If T is 'bounds', returns a list [Tmin, Tmax] which are the temperature bounds.
            If T is a float, 
                if der = 0 then returns entropy change in ideal gas state from reference state 100KPa, 25 degree centigrade to
                arbitrary T and and 100KPa
                if der = 1 then returns first derivative wrt T
                if der = 2 returns second derivative 
                else raises ValueError
        EntropyIdealGasReference:: Sig(T) = integral(Cp/T, Tref, T)
            Ideal gas entropy at any other pressure is EntropyIdealGasReference(T) - R*ln(P/Preference)               

        '''
        if T == 'bounds':
            return [self.__EntropyIdealGasReference.Tmin, self.__EntropyIdealGasReference.Tmax]
        else:
            if der == 0:
                return self.__EntropyIdealGasReference.Evaluate(T)
            elif der == 1:
                return self.__EntropyIdealGasReference.Evaluate.derivatives(T)[1]
            elif der == 2:
                return self.__EntropyIdealGasReference.Evaluate.derivatives(T)[2]
            else:
                raise ValueError('der values must be in [0,1,2]')
                
    def GibbsEnergyIdealGasReference(self, T, der):
        '''
        
            If T is 'bounds', returns a list [Tmin, Tmax] which are the temperature bounds.
            If T is a float, 
                if der == 0, then returns Gibbs Energy in ideal gas state at reference pressure Preference = 101,325 Pa in J/kmol
                if der == 1, returns the first derivative
                if der == 2, returns the second derivative
                else raises ValueError               
            Ideal gas Gibbs Energy at any other pressure is GibbsEnergyIdealGasReference(T) + R*T*ln(P/Preference)
        GibbsEnergyIdealGasReference:: Gig(T) = Gfig + Hig(T) - Hfig - T*(Sig(T) - Sreference)
        '''
        if T == 'bounds':
            return [self.__GibbsEnergyIdealGasReference.Tmin, self.__GibbsEnergyIdealGasReference.Tmax]
        else:
            if der == 0:
                return self.__GibbsEnergyIdealGasReference.Evaluate(T)
            elif der == 1:
                return self.__GibbsEnergyIdealGasReference.derivatives(T)[1]
            elif der == 2:
                return self.__GibbsEnergyIdealGasReference.derivatives(T)[2]
            else:
                raise ValueError('der values must be in [0,1,2]')


    def __CorrelationBank(self, property, T):
        '''
        Evaluates the required correlation at the temperature T (in K).
        The property to be calculated is passed through the string property.
        The property string could be:
            'VapourPressure'
            'LiquidDensity'
            'EnthalpyVapourization'
            'SpecificHeatCapacity'
        '''
        if property not in ['VapourPressure',
                            'LiquidDensity',
                            'EnthalpyVapourization',
                            'SpecificHeatCapacity']:
            raise ValueError('Can\'t evaluate property.  The name '+str(property)+' means nothing.')
        else:
            if property == 'VapourPressure':
                c1 = self.__listVapourPressureParameters[0]
                c2 = self.__listVapourPressureParameters[1]
                c3 = self.__listVapourPressureParameters[2]
                c4 = self.__listVapourPressureParameters[3]
                c5 = self.__listVapourPressureParameters[4]
                vp = scipy.exp(c1 + c2/T + c3*scipy.log(T) + c4*T**c5)
                return vp 
            elif property == 'LiquidDensity':
                c1 = self.__listLiquidDensityParameters[0]
                c2 = self.__listLiquidDensityParameters[1]
                c3 = self.__listLiquidDensityParameters[2]
                c4 = self.__listLiquidDensityParameters[3]
                dens = c1/c2**(1+(1-T/c3)**c4)
                return dens         
            elif property == 'EnthalpyVapourization': 
                c1 = self.__listEnthalpyVapourizationParameters[0]
                c2 = self.__listEnthalpyVapourizationParameters[1]
                c3 = self.__listEnthalpyVapourizationParameters[2]
                c4 = self.__listEnthalpyVapourizationParameters[3]
                Tr = T/self.CriticalTemperature
                hvap = c1*(1-Tr)**(c2+c3*Tr+c4*Tr*Tr)  
                return hvap
            elif property == 'SpecificHeatCapacity':
                c1 = self.__listSpecificHeatCapacityParameters[0]
                c2 = self.__listSpecificHeatCapacityParameters[1]
                c3 = self.__listSpecificHeatCapacityParameters[2]
                c4 = self.__listSpecificHeatCapacityParameters[3]
                c5 = self.__listSpecificHeatCapacityParameters[4]
                csin = c3/T
                ccos = c5/T
                cp = c1 + c2*(csin/scipy.sinh(csin))**2+c4*(ccos/scipy.cosh(ccos))**2  
                return cp
    def __readCorrelation(self, data, idata):    
        info = data[idata].split()
        property = info[0]
        nparams = int(info[1])
        moreinfo = data[idata+1].split('units:')[0].split()
        self.listparams = []
        for iparams in range(nparams):
            self.listparams.append(float(moreinfo[iparams]))
        if len(self.listparams) == nparams:
            mindata = data[idata+2].split('units:')[0].split()
            Tmin = float(mindata[0])
            Valmin = float(mindata[1])
            maxdata = data[idata+3].split('units:')[0].split()
            Tmax = float(maxdata[0])
            Valmax = float(maxdata[1])
            self.listparams += [Tmin, Tmax]
            if property == 'VapourPressure':
                self.__listVapourPressureParameters = copy.copy(self.listparams)
            elif property == 'LiquidDensity':
                self.__listLiquidDensityParameters = copy.copy(self.listparams)
            elif property == 'EnthalpyVapourization':
                self.__listEnthalpyVapourizationParameters = copy.copy(self.listparams)
            elif property == 'SpecificHeatCapacity':
                self.__listSpecificHeatCapacityParameters = copy.copy(self.listparams)
            Vmin = self.__CorrelationBank(property, Tmin)
            Vmax = self.__CorrelationBank(property, Tmax)
            dVmin = abs(Vmin - Valmin)/(Valmin+1e-6) * 100
            dVmax = abs(Vmax - Valmax)/(Valmax+1e-6) * 100
            if dVmin < 1 and dVmax < 1:
                return True
            else:
                print 'Property '+property+' correlation does not match end points.'
                return False                        
        else:
            print 'Something wrong with data for '+property+' in file '+self.Name+'.txt.  The number of parameters don\'t match the number stated.'
            raise ValueError()         
    def __readScalar(self,info):
        scalar = info[0]
        try:
            value = float(info[1])
            if scalar == 'MolecularWeight':
                self.MolecularWeight = value
            elif scalar == 'NormalBoilingPoint':
                self.NormalBoilingPoint = value
            elif scalar == 'FreezingPoint':
                self.FreezingPoint = value
            elif scalar == 'CriticalTemperature':
                self.CriticalTemperature = value
            elif scalar == 'CriticalPressure':
                self.CriticalPressure = value
            elif scalar == 'CriticalVolume':
                self.CriticalVolume = value
            elif scalar == 'AcentricFactor':
                self.AcentricFactor = value
            elif scalar == 'ReferenceTemperature':
                self.ReferenceTemperature = value
            elif scalar == 'ReferencePressure':
                self.ReferencePressure = value
            elif scalar == 'EnthalpyFormation':
                self.EnthalpyFormation = value
            elif scalar == 'GibbsEnergyFormation':
                self.GibbsEnergyFormation = value
            elif scalar == 'EntropyReference':
                self.EntropyReference = value
            return True
        except ValueError:
            print 'Could not read value for '+scalar+'.'
            return False
    def __readDataFile(self):
        '''
        Directs the code to read the datafile called [name].txt and set the various attributes of the object. 
        '''
        filename = self.Name+'.txt'
        if os.path.isfile(filename):
            f = open(filename,'r')
            data = f.readlines()
            f.close()            
            putname = data[0].split()[0]  #The name is the very first bit of text in the file
            if putname != self.Name:
                print 'There is something wrong in '+filename+'. Does not match with '+putname+'.'
                raise NameError()
            else:                
                listscalar = ['MolecularWeight',
                            'NormalBoilingPoint',
                            'FreezingPoint',
                            'CriticalTemperature',
                            'CriticalPressure',
                            'CriticalVolume',
                            'AcentricFactor',
                            'ReferenceTemperature',
                            'EnthalpyFormation',
                            'GibbsEnergyFormation',
                            'EntropyReference']
                listcorrelation = ['VapourPressure',
                                    'LiquidDensity',
                                    'EnthalpyVapourization',
                                    'SpecificHeatCapacity']
                for idata in range(len(data)):
                    if idata > 0:
                        info = data[idata]    
                        info = info.split()
                        property = info[0]
                        if property in listscalar:
                            if self.__readScalar(info):
                                junk = listscalar.pop(listscalar.index(property))                   
                        elif property in listcorrelation:
                            if self.__readCorrelation(data, idata):  
                                junk = listcorrelation.pop(listcorrelation.index(property))
                                
                if len(listscalar) > 0:
                    print 'The following scalar parameters were not available in '+filename
                    for par in listscalar:
                        print par
                    raise IndexError('Not all scalar data could be read.')
                if len(listcorrelation) > 0:
                    print 'The following correlation parameters could not be read from '+filename
                    for par in listcorrelation:
                        print par
                    raise IndexError('Not all correlation parameters could be read.')
        else:
            print 'Filename '+filename+' does not exist.'


class Molecule_CubicEquationOfState(Molecule):
    '''
    
     The Molecule_CubicEquationOfState class of objects represent pure component  properties of molecules calculated using the specified cubic Equation Of State.
    
    The 'Molecule_CubicEquationOfState' class inherits from the 'Molecule' parent class.  
    The class requires the molecule name and the equation of state name: either 'SRK' for Soave-Redlich-Kwong or 'PR' for Peng-Robinson or 'Modified PR' for the Modified Peng - Robinson equation or 'PRSV1' for the Peng-Robinson-Stryek-Vera Equation 1 or 'PRSV2' for the Peng-Robinson-Stryek-Vera Equation 2.

    A cubic equation of state relates Pressure (P) to temperature (T) and molar volume (V)
    Its general form is:: P = RT/(V-b) - a(T)/[(V+epsilon*b)*(V+sigma*b)]
                            a(T) = OmegaA*alpha(T)*R**2*Tc**2/Pc; where alpha(T) = [1 + (0.48000 + 1.57400*w - 0.17600*w**2)*(1-Tr**0.5)]**2 ... for SRK equation of state
                                                                                 = [1 + (0.37464 + 1.54226*w - 0.26992*w**2)*(1-Tr**0.5)]**2 ... for PR  equation of state
                            b    = OmegaB*R*Tc/Pc
                            w    = AcentricFactor
                            Tc   = Critical Temperature
                            Pc   = Critical Pressure
                            Tr   = T/Tc = Reduced Temperature
                            The following values for SRK and PR Equations of State are also used:
                            PARAMETER           VALUE for SRK           VALUE for PR    VALUE for PRSV1  VALUE FOR PRSV2  VALUE FOR Modified PR
                                epsilon             0                       -0.414214   -0.414214         -0.414214        -0.414214 
                                sigma               1                        2.414214    2.414214          2.414214         2.414214
                                OmegaA              0.42748                  0.457235    0.457235          0.457235         0.457235
                                OmegaB              0.08664                  0.077796    0.077796          0.077796         0.077796
                                c                   0.69315                  0.62323     0.62323           0.62323          0.62323
    The equation of state can also be cast in the form of a cubic equation in the compressibility Z = PV/RT
                            Z**3 + A*Z**2 + B*Z + C = 0
                                where,
                                    A    = +pp*[epsilon + sigma - ppp]
                                    B    = +pp**2*[ap + sigma*epsilon - (sigma+epsilon)*ppp]
                                    C    = -pp**3*[ap + sigma*epsilon*ppp]
                                    pp   = P*b/(R*T)
                                    ap   = a/(P*b**2)
                                    ppp  = (1+pp)/pp)
    ***************************************************************************
    The following private utility methods are available
    ***************************************************************************
    __setParameters             : Uses the critical constants to set the parameters of the cubic equation of state specified.
                                    Raises ValueError if the equation of state is not ['SRK','PR']
    __a                         : The a(T) parameter of the equation of state from the equation of state correlations.
    __getlnphi                  : Returns logarithm of the fugacity coefficient as a function of Z, T and P
    ***************************************************************************
    The following public methods are available
    ***************************************************************************
    getfugacoeff()                : Returns the fugacity coefficient as a function of Z, T and P
    CompressibilityFactors()      : Returns vapour and liquid compressibility factors for a given temperature (K) and pressure (Pa).
    ResidualGibbsEnergy()        : Returns the difference between Gibbs energies of ideal-gas and real gas at given T (K) and P(Pa)
    a()                           : Returns the paramater a(T) needed in Cubic Equations of State
    ResidualEnthalpy          : Returns the departure function for Enthalpy at a given T(K) and P(Pa)
    RealEnthalpy               : Returns the Enthalpy of the real fluid with reference state being an ideal gas at 25 degree centigrade and 1 bar pressure, for which
    H=0
    ResidualEntropy            : Returns Entropy departure function for the real fluid at a given T (K) and P(Pa)
    RealEntropy                : Returns real fluid entropy with reference to S=0 for an ideal gas at 25 degree centigrade and 1 bar pressure.
    PVisotherms                 : Gives isothermal PV plots and the PV saturation curve
    PHisotherms                 : Gives isothermal PH plots and the PH saturation curve
    TSisobars                   : Gives isobaric   TS plots and the TS saturation curve
    Whatsthetemperature         : This function is a crude equation solver used to generate saturation temperature values for the TS isobars.
    VapourPressureEOS           : Returns the Saturation Pressure for a given Temperature from the given Equation of State
    HvapEOS                     : Returns the Enthalpy of Vapourisation for a given Temperature and Pressure from given Cubic Equation of State
    VlsatEOS                    : Returns the Saturated Liquid Phase Volumes for a given Temperature and Saturation Pressure using the Cubic Equation of State
    adjustparams                : Returns optimum value of parameters of given Cubic Equation of States - Modified PR, PRSV1,PRSV2, using a Least Squares Approximation and also plots of Saturation Pressure,
                                  Enthalpy of Vapourization and Saturated Liquid Phase Volume for both adjusted EOS and correlated values.
    plotEOS                     : Returns plots of Vapour Pressure, Enthalpy of Vapourization,Liquid Phase Saturated Volume with Temperature using unoptimized Equation of State and
                                  actual correlated values. 
    '''    
    def __init__(self, moleculename, equationofstatename):
        '''
        __init__(self, moleculename, equationofstatename):
            moleculename        = name of the molecule.  Is a string.  Molecule data is in [moleculename].txt
            equationofstatename = name of the equation of state.  Is a string: either 'SRK' or 'PR'
        The __init__ function also initializes the parent class which is Molecule with:
            Molecule.__init__(self, moleculename)
        '''
        Molecule.__init__(self, moleculename)
        self.R                      = 8314.0 #J/kmol-K
        self.EquationOfState        = equationofstatename
        
        self.OmegaA                 = 'junk'
        self.OmegaB                 = 'junk'
        self.epsilon                = 'junk' 
        self.sigma                  = 'junk'
        self.c                      = 'junk' 
        self.b                      = 'junk'
        if self.EquationOfState=='PRSV2' or self.EquationOfState=='Modified PR':
            self.k1 = 0.0
            self.k2 = 0.0
            self.k3 = 0.0
        elif self.EquationOfState=='PRSV1':
            self.k1=0.0
          
        self.__a                    = 'junk'
        self.__setParameters()
    def a(self, T,der):
        '''
        a(T)
            INPUTS
                T   = 'bounds'
                    OR
                T   = Scalar. Temperature in K
            OUTPUTS
                if T == 'bounds':
                    returns [Tmin, Tmax] which are the temperature bounds for a.
                else:
                    returns a - the Cubic Equation of State a parameter
        '''
        if T == 'bounds':
            return [self.__a.Tmin, self.__a.Tmax]
        elif der==0:
            return self.__a.Evaluate(T)
        elif der==1:
            return self.__a.Evaluate.derivatives(T)[1]
        else:
            raise ValueError('der values must be in [0,1,2]')
        
    def __setParameters(self):
        R = self.R
        Pc = self.CriticalPressure
        Tc = self.CriticalTemperature
        w  = self.AcentricFactor
        if self.EquationOfState == 'PR': 
            self.epsilon    = -0.414214
            self.sigma      = 2.414214
            self.OmegaA     = 0.457235
            self.OmegaB     = 0.077796
            self.c          = 0.62323
            self.b          = self.OmegaB*R*Tc/Pc
            alpha = lambda T: (1 + (0.37464 + 1.54226*w - 0.26992*w**2)*(1-(T/Tc)**0.5))**2 
            Tmin = 100.0            #Random values
            Tmax = 2000.0           #Random values
            xx = scipy.linspace(Tmin, Tmax, 2000)
            yy = self.OmegaA*alpha(xx)*R**2*Tc**2/Pc
            self.__a = PureTemperatureProperty()
            self.__a.Tmin = Tmin
            self.__a.Tmax = Tmax
            self.__a.Evaluate = scipy.interpolate.UnivariateSpline(xx, yy)
        elif self.EquationOfState == 'SRK':
            self.epsilon    = 0.0
            self.sigma      = 1.0
            self.OmegaA     = 0.42748 
            self.OmegaB     = 0.08664
            self.c          = 0.69315
            self.b          = self.OmegaB*R*Tc/Pc
            alpha = lambda T: (1 + (0.48000 + 1.57400*w - 0.17600*w**2)*(1-(T/Tc)**0.5))**2 
            Tmin = 100.0          #Random values
            Tmax = 2000.0         #Random values
            xx = scipy.linspace(Tmin, Tmax, 2000)
            yy = self.OmegaA*alpha(xx)*R**2*Tc**2/Pc
            self.__a = PureTemperatureProperty()
            self.__a.Tmin = Tmin
            self.__a.Tmax = Tmax
            self.__a.Evaluate = scipy.interpolate.UnivariateSpline(xx, yy)
        
        
        elif self.EquationOfState == 'PRSV2': 
            self.epsilon    = -0.414214
            self.sigma      = 2.414214
            self.OmegaA     = 0.457235
            self.OmegaB     = 0.077796
            self.c          = 0.62323
            k1=self.k1
            k2=self.k2
            k3=self.k3
            self.b          = self.OmegaB*R*Tc/Pc
#            alpha= lambda T:(1+(T/Tc)**0.5)*(0.7-T/Tc)*(self.k1+self.k2*(self.k3-T/Tc)*(1-(T/Tc)**0.5))+0.378893 + 1.4897153*w - 0.17131848*w**2+0.0196554*w**3
            alpha= lambda T:(1 + (0.378893 + 1.4897153*w - 0.17131848*w**2 + 0.0196554*w**3 + (1+(T/Tc)**0.5)*(0.7-T/Tc)*(k1+k2*(k3-T/Tc)*(1-(T/Tc)**0.5)))*(1-(T/Tc)**0.5))**2 
            Tmin = 100.0           #Random values
            Tmax = 2000.0           #Random values
            xx = scipy.linspace(Tmin, Tmax, 2000)
            yy = self.OmegaA*alpha(xx)*R**2*Tc**2/Pc
            self.__a = PureTemperatureProperty()
            self.__a.Tmin = Tmin
            self.__a.Tmax = Tmax
            self.__a.Evaluate = scipy.interpolate.UnivariateSpline(xx, yy) 

        elif self.EquationOfState == 'PRSV1': 
            self.epsilon    = -0.414214
            self.sigma      = 2.414214
            self.OmegaA     = 0.457235
            self.OmegaB     = 0.077796
            self.c          = 0.62323
            k1=self.k1
            self.b          = self.OmegaB*R*Tc/Pc
#            alpha= lambda T:(1+(T/Tc)**0.5)*(0.7-T/Tc)*(self.k1+self.k2*(self.k3-T/Tc)*(1-(T/Tc)**0.5))+0.378893 + 1.4897153*w - 0.17131848*w**2+0.0196554*w**3
            alpha= lambda T:(1 + (0.378893 + 1.4897153*w - 0.17131848*w**2 + 0.0196554*w**3 + (1+(T/Tc)**0.5)*(0.7-T/Tc)*(k1))*(1-(T/Tc)**0.5))**2 
            Tmin = 100.0           #Random values
            Tmax = 2000.0           #Random values
            xx = scipy.linspace(Tmin, Tmax, 2000)
            yy = self.OmegaA*alpha(xx)*R**2*Tc**2/Pc
            self.__a = PureTemperatureProperty()
            self.__a.Tmin = Tmin
            self.__a.Tmax = Tmax
            self.__a.Evaluate = scipy.interpolate.UnivariateSpline(xx, yy) 
        
        elif self.EquationOfState == 'Modified PR': 
            self.epsilon    = -0.414214
            self.sigma      = 2.414214
            self.OmegaA     = 0.457235
            self.OmegaB     = 0.077796
            self.c          = 0.62323
            k1=self.k1
            k2=self.k2
            k3=self.k3
            self.b          = self.OmegaB*R*Tc/Pc
#            alpha= lambda T:(1+(T/Tc)**0.5)*(0.7-T/Tc)*(self.k1+self.k2*(self.k3-T/Tc)*(1-(T/Tc)**0.5))+0.378893 + 1.4897153*w - 0.17131848*w**2+0.0196554*w**3
            alpha= lambda T:(1 + (0.37464 + 1.54226*w - 0.26992*w**2 + (1+(T/Tc)**0.5)*(0.7-T/Tc)*(k1+k2*(k3-T/Tc)*(1-(T/Tc)**0.5)))*(1-(T/Tc)**0.5))**2 
            Tmin = 100.0           #Random values
            Tmax = 2000.0           #Random values
            xx = scipy.linspace(Tmin, Tmax, 2000)
            yy = self.OmegaA*alpha(xx)*R**2*Tc**2/Pc
            self.__a = PureTemperatureProperty()
            self.__a.Tmin = Tmin
            self.__a.Tmax = Tmax
            self.__a.Evaluate = scipy.interpolate.UnivariateSpline(xx, yy) 

        else:
            print 'ERROR! Equation of State '+str(self.EquationOfState)+' is unknown.'
            raise ValueError('Equation of State not recognized.') 
    def __getlnphi(self, Z, T, P):
        '''
        Private function.
            INPUTS
                Z       = compressibility factor
                T       = Temperature (K)
                P       = Pressure (Pa)
            OUTPUT
                lnphi   = logarithm of the fugacity coefficient.
                        = Z - 1 - ln(Z-pp) + ap*pp/(sigma-epsilon)*ln[(Z+pp*epsilon)/(Z+pp*sigma)]
                    where,
                            pp  =   P*b/(R*T)
                            ap  =   a/(P*b**2)
        '''
        b               =   self.b
        a               =   self.__a.Evaluate(T)
        R               =   self.R
        sigma           =   self.sigma
        epsilon         =   self.epsilon
        pp              =   P*b/(R*T)
        ap              =   a/(P*b**2)
        lnphi   =   Z - 1.0 - scipy.log(Z - pp) + ap*pp/(sigma - epsilon)*scipy.log((Z+pp*epsilon)/(Z+pp*sigma))
        
        return lnphi
    
    def getfugacoeff(self,Z,T,P):
        
        '''
        Public function.
            INPUTS
                Z       = compressibility factor
                T       = Temperature (K)
                P       = Pressure (Pa)
            OUTPUT
                phi   = fugacity coefficient.
                        = exp(Z - 1 - ln(Z-pp) + ap*pp/(sigma-epsilon)*ln[(Z+pp*epsilon)/(Z+pp*sigma)])
                    where,
                            pp  =   P*b/(R*T)
                            ap  =   a/(P*b**2)
        '''
        
        
        b               =   self.b
        a               =   self.__a.Evaluate(T)
        R               =   self.R
        sigma           =   self.sigma
        epsilon         =   self.epsilon
        pp              =   P*b/(R*T)
        ap              =   a/(P*b**2)
        lnphi   =   Z - 1.0 - scipy.log(Z - pp) + ap*pp/(sigma - epsilon)*scipy.log((Z+pp*epsilon)/(Z+pp*sigma))
        phi=scipy.exp(lnphi)
        return phi
        
        
        
    def CompressibilityFactors(self, T, P):
        '''
        CompressibilityFactors(T, P):
            INPUTS
                T       = Scalar: Temperature in K.
                P       = Scalar: Pressure in Pa
            OUTPUTS
                ZG, ZL
                    ZG  = Vapour/Gas Compressibility Factor
                    ZL  = Liquid Compressibility Factor
                If T > CriticalTemperature, ZG = ZL.                
        Z = PV/RT        
        The compressibility factors are obtained by solving the following cubic:
            Z**3 + A*Z**2 + B*Z + C = 0
                where,
                    A    =  pp*(epsilon + sigma - ppp)
                    B    =  pp**2*[ap + epsilon*sigma - (epsilon+sigma)*ppp]
                    C    = -pp**3*[ap + epsilon*sigma*ppp]
                    ap   =  a/(P*b**2)
                    pp   =  P*b/(R*T)
                    ppp  =  (1 + pp)/pp
        If T < Critical Temperature, there are 3 real roots.  The maximum is ZG, minimum is ZL.  Middle root is discarded.
        If T > Critical Temperature, there is  only 1 real root and ZG = ZL.
        '''
        R       = self.R
        epsilon = self.epsilon
        sigma   = self.sigma
        b       = self.b
        a       = self.a(T,0)
        pp   = P*b/(R*T)
        ppp  = (1.0 + pp)/pp
        ap   = a/(P*b**2)
        A =  pp*(epsilon + sigma - ppp)
        B =  pp**2*(ap + sigma*epsilon - (epsilon + sigma)*ppp)
        C = -pp**3*(ap + sigma*epsilon*ppp) 
        compressibilities = Utilities.cubicsolver([A, B, C])
        ZG=max(compressibilities)
        ZL=min(compressibilities)
        return ZG,ZL
    def ResidualGibbsEnergy(self, T,P):    
        '''
        ResidualGibbsEnergy(T, P):
            INPUTS
                T = 'bounds'
                    OR
                T   = Scalar. Temperature in K
                P   = Scalar. Pressure in Pa
                
            OUTPUTS
                If T == 'bounds':
                    [Tmin, Tmax]
                else:
                    GR  = Residual Gibbs Energy for Pure Component
        Hence,
            GR = ResidualGibbsEnergy(T, P)
                OR
            [Tmin, Tmax] = ResidualGibbsEnergy('bounds',junk)

        The fugacity coefficient 'phi' is a measure of the residual Gibbs energy of the system due to non-ideal behaviour.
        GR  =   G - Gig = RT*ln(phi)
                where,
                    Gig = Gfig + (H - Hfig) - T*(S - Sreference)@(Pref=101,325Pa) + R*T*ln(P/Pref) 
                        = Gig@Pref + R*T*ln(P/Pref)
                        = self.GibbsEnergyIdealGasReference(T) + R*T*ln(P/Pref)
        If P < Psat, the fugacity coefficient is given by:
            ap      = a/(P*b**2)
            pp      = P*b/(R*T)
            ln(phi) = Z - 1 - ln(Z-pp) + ap*pp/(sigma-epsilon)*ln[(Z+pp*epsilon)/(Z+pp*sigma)]
        If P > Psat, the fugacity coefficient is given by:
            ln(phi) = ln(phi_Psat) + VL(P-Psat)/RT - ln(P/Psat)
        GR = R*T*ln(phi)
        '''
        [Tpmin, Tpmax] = self.VapourPressure('bounds',0)
        [Tvmin, Tvmax] = self.LiquidDensity('bounds',0)
        Tmin = max([Tpmin, Tvmin])
        Tmax = min([Tpmax, Tvmax])
        if T == 'bounds':
            return [Tmin, Tmax]
        else:
            R = self.R
            if T >= Tmin and T <= Tmax:
                Psat    = self.VapourPressure(T,0)
                if P <= Psat:
                    ZG, ZL = self.CompressibilityFactors(P, T)
                    logphi = self.__getlnphi(ZG, T, P)
                else:
                    ZGsat, ZLsat = self.CompressibilityFactors(Psat, T)
                    VL = 1.0/self.LiquidDensity(T,0)
                    logphi_sat = self.__getlnphi(ZGsat, T, Psat)
                    poynting = VL*(P - Psat)/(R*T) - scipy.log(P/Psat)
                    logphi = logphi_sat + poynting
                return R*T*logphi
            else:
                print 'Temperature '+str(T)+'K is not in temperature bounds: '+str(Tmin)+'K to '+str(Tmax)+'K.'            
                raise ValueError('ERROR!: Temperature bounds exceeded for pure component.')
    def ResidualEnthalpy(self,T,P,state):
        '''
          ResidualEnthalpy(self,T,P,state):
            INPUTS
                
                T   = Scalar. Temperature in K
                P   = Scalar. Pressure in Pa
                state='liquid' OR 'vapour'
            OUTPUTS
               HR  = Residual Gibbs Energy for Pure Component in J/kmol
        Hence,
            HR = ResidualGibbsEnergy(T, P)
            HR = R*T*(Z-1)+(T*da/dT-a)/((sigma-epsilon)*b)*(ln((Z+sigma*pp)/(Z+epsilon*pp)))
            pp=Pb/RT
            where Z=ZL if 'liquid'
            Z=ZG if 'vapour'
            ZG,ZL are obtained from self.CompressibilityFactors(T,P)
        '''    
       
        R=self.R
        B=(self.b)*P/(R*T)
        if state=='liquid':
            z=float(self.CompressibilityFactors(T,P)[1])
        elif state=='vapour':
            z=float(self.CompressibilityFactors(T,P)[0])
        else:
            raise ValueError('wrong words')
        epsilon=self.epsilon
        sigma=self.sigma
        A=(T*self.a(T,1)-self.a(T,0))/((sigma-epsilon)*(self.b))
        C=scipy.log(z+sigma*B)-scipy.log(z+epsilon*B)
        dh=R*T*(z-1)+A*C    
        return dh
  
    def RealEnthalpy(self,T,P):
        '''
        Real Enthalpy(self,T,P)
        INPUTS
          T = Scalar.Temperature in K
          P = Scalar.Pressure in Pa
        OUTPUTS
          Enthalpy H in J/kmol
        
        Reference Thermodynamic State is ideal gas at 298.16 K and 100000 Pa
        
        H(T,P)-Hig(298K,100000Pa)=(Hig(T,P)-Hig(298K,100000Pa))+HR(T,P))
        
        Therefore, Real Enthalpy=HR(T,P)+integral(cpig)dT(298K,T)
        
        Enthalpy difference w.r.t. Reference Thermodynamic State is obtained
        '''
        [Tpmin, Tpmax] = self.VapourPressure('bounds',0)
        [Tvmin, Tvmax] = self.SpecificHeatCapacity('bounds',0)
        Tmin = max([Tpmin, Tvmin])
        Tmax = min([Tpmax, Tvmax])
        if T == 'bounds':
            return [Tmin, Tmax]
        else:
             if T >= Tmin and T <= Tmax:
                Psat=self.VapourPressure(T,0)
                if P>=Psat:
                 rh=self.ResidualEnthalpy(T,P,'liquid')+self.EnthalpyIdealGas(T,0)
                elif P<Psat:
                 rh=self.ResidualEnthalpy(T,P,'vapour')+self.EnthalpyIdealGas(T,0)
                return rh
             else:
                print 'Temperature '+str(T)+'K is not in temperature bounds: '+str(Tmin)+'K to '+str(Tmax)+'K.'            
                raise ValueError('ERROR!: Temperature bounds exceeded for pure component.')
    def ResidualEntropy(self,T,P,state):
        '''
         ResidualEntropy(self,T,P,state):
            INPUTS
                
                T   = Scalar. Temperature in K
                P   = Scalar. Pressure in Pa
                state='liquid' OR 'vapour'
            OUTPUTS
               SR  = Residual Entropy for Pure Component in J/kmolK
         Hence,
            HR = Residual Enthalpy
            HR = R(Z-pp)+(da/dT)/((sigma-epsilon)*b)*(ln((Z+sigma*pp)/(Z+epsilon*pp)))
            where Z=ZL if 'liquid'
            Z=ZG if 'vapour'
            pp=Pb/RT
            ZG,ZL are obtained from self.CompressibilityFactors(T,P)
        '''    
         
        R=self.R
        B=(self.b)*P/(R*T)
        sigma=self.sigma
        epsilon=self.epsilon
        if state=='liquid':
            z=float(self.CompressibilityFactors(T,P)[1])
        elif state=='vapour':
            z=float(self.CompressibilityFactors(T,P)[0])
        else:
            raise ValueError('wrong words')
        A=self.a(T,1)/((2*1.414214)*(self.b))
        C=scipy.log(z+sigma*B)-scipy.log(z+epsilon*B)
        ds=R*(z-B)+A*C
        return ds
    def RealEntropy(self,T,P):
        '''
        RealEntropy(self,T,P)
        
        INPUTS
        T = Scalar.Temperature in K.
        P = Scalar.Pressure in Pa.
        
        Reference Thermodynamic State is ideal gas at 298K and 100000 Pa.
        
        Real Entropy S(T,P)-Sig(298K,100000)=(Sig(T,P)-Sig(298K,100000))+SR(T,P))
        
        Therefore, Real Entropy =SR(T,P)+integral(cpig/T)dT(25deg,T)-R*ln(P/1bar)
        
        Entropy Difference is obtained w.r.t. Reference Thermodynamic State
        '''
        [Tpmin, Tpmax] = self.VapourPressure('bounds',0)
        [Tvmin, Tvmax] = self.SpecificHeatCapacity('bounds',0)
        Tmin = max([Tpmin, Tvmin])
        Tmax = min([Tpmax, Tvmax])
        if T == 'bounds':
            return [Tmin, Tmax]
        else:
            if T>=Tmin and T<=Tmax and P<=self.CriticalPressure:
                    Tsat=self.Whatsthetemperature(P)
                    if T<=Tsat:
                     rs=self.ResidualEntropy(T,P,'liquid')+self.EntropyIdealGasReference(T,0)-(self.R)*scipy.log(P/100000.0)
                    elif T>Tsat:
                     rs=self.ResidualEntropy(T,P,'vapour')+self.EntropyIdealGasReference(T,0)-(self.R)*scipy.log(P/100000.0)
                    return rs
            else:
                print 'Temperature '+str(T)+'K is not in temperature bounds: '+str(Tmin)+'K to '+str(Tmax)+'K,or Pressure greater than critical pressure'           
                raise ValueError('ERROR!: Temperature bounds exceeded for pure component OR Pressure greater than critical Pressure')
  
    def Whatsthetemperature(self,value):
        '''
        INPUTS
        Value = Scalar.Pressure in Pa.
        OUTPUTS
        T in K = Temperature corresponding to a VapourPressure = Value Pa.
        
        The function evaluates the Temperature using the fsolve method in scipy module
        function ff(T):
        y= VapourPressure at Temperature T
        error=scipy.fabs(value-y)
        return error
        
        the Temperature is obtained as T=scipy.optimize.fsolve(ff,guess)
        
        Guess: Guess Temperature in K
        
        ''' 
        if value>self.CriticalPressure:
             raise ValueError('pressure not in range')
        else:
             guess=self.CriticalTemperature
             def ff(T):
                 y=self.VapourPressure(T,0)
                 error=(value-y)
                 return error

             xxx=scipy.optimize.fsolve(ff,guess)
             return xxx
         
    def TSisobars(self,Prange):
        '''
        INPUTS
        Prange = [] list of Pressures in Pa
        OUTPUTS
        1. Saturation Curve:
        
        Generates the Saturation T-S envelope for a range of Temperatures in K.
        
        Range = [Tmin,Tmax,1000 data points]
        Tmin = Reference Temperature in K
        Tmax = Critical Temperature in K
        for some T in Range
        Compute Vapour Pressure Psat in Pa
        Compute <RealEntropy (T,Psat)> for all T in Range
        Plot a graph of Range vs.  <RealEntropy (T,Psat)> 
        
        2. Isobars
        
        Generates constant pressure T-S curves for all Pressures in Prange
        
        For a P in Prange 
        
        Compute Whatsthetemperature(P) = Tsat in K
         For a T in the Range of Temperature : 
         Compute <RealEntropy 'Vapour'(T,P)> if T>Tsat
         <RealEntropy 'Liquid'(T,P)> if T<=Tsat
         Plot a graph of Range vs. RealEntropy values.
       
        
        '''
        
        Tmax1=self.CriticalTemperature 
        Tmin1=self.ReferenceTemperature
        Pc=self.CriticalPressure
        Tc=self.CriticalTemperature
        TT=scipy.linspace(Tmin1,Tmax1,100)
        Tsat=[]
        SSL=[]
        SSG=[]
        for Tsat1 in TT:
                    P=self.VapourPressure(Tsat1,0)
                    SSL.append(self.ResidualEntropy(Tsat1,P,'liquid')+self.EntropyIdealGasReference(Tsat1,0)-(self.R)*scipy.log(P/100000.0))
                    SSG.append(self.ResidualEntropy(Tsat1,P,'vapour')+self.EntropyIdealGasReference(Tsat1,0)-(self.R)*scipy.log(P/100000.0))
                    Tsat.append(Tsat1)
        Tsat.append(Tc)
        SSL.append(self.ResidualEntropy(Tc,Pc,'liquid')+self.EntropyIdealGasReference(Tc,0)-(self.R)*scipy.log(P/100000.0))
        SSG.append(self.ResidualEntropy(Tc,Pc,'liquid')+self.EntropyIdealGasReference(Tc,0)-(self.R)*scipy.log(P/100000.0))
        
          
        for P in Prange:
            Trange=scipy.linspace(Tmin1,Tmax1,100)
            S=[]
            for T in Trange:
                S.append(self.RealEntropy(T,P))
            pylab.plot(S,Trange,'b')
        
        pylab.plot(SSL,Tsat,'r')   
        pylab.plot(SSG,Tsat,'r')
        pylab.show()
        pylab.clf()
        
        
        
    
    def PHisotherms(self,Trange):
        '''
       INPUTS
        Trange = [] list of Temperatures in K.
        OUTPUTS
        1. Saturation Curve:
        
        Generates the Saturation P-H envelope for a range of Temperatures in K.
        
        Range = [Tmin,Tmax,1000 data points]
        Tmin = Refernce Temperature in K
        Tmax = Critical Temperature in K
        for some T in Range
        Compute Vapour Pressure Psat in Pa
        Compute <RealEnthalpy (T,Psat)> for all T in Range
        Plot a graph of Vapour Pressure vs.  <RealEnthalpy (T,Psat)> 
        
        2. Isotherms
        
        Generates constant temperature P-H curves for all Temperatures in Trange
        
        For a T in Trange 
        
        Compute Vapour Pressure = Psat in Pa
         For a P in the Range of Pressure : 
         Compute <RealEnthalpy 'Vapour'(T,P)> if P<=Psat
         <RealEnthalpy 'Liquid'(T,P)> if P>Psat
         Plot a graph of Range vs. <RealEnthalpy> values.
        
        '''
        Tmax = self.CriticalTemperature
        Tmin =self.ReferenceTemperature
        Tc=self.CriticalTemperature
        Pc=self.CriticalPressure
        TT =scipy.linspace(Tmin,Tmax,1000)
        Psat=[]
        HHL=[]
        HHG=[]
        for T in TT:
                Psat1=float(self.VapourPressure(T,0))
                HHL.append(self.ResidualEnthalpy(T,Psat1,'liquid')+self.EnthalpyIdealGas(T,0))
                HHG.append(self.ResidualEnthalpy(T,Psat1,'vapour')+self.EnthalpyIdealGas(T,0))
                Psat.append(float(self.VapourPressure(T,0)))
           
        Psat.append(self.CriticalPressure)
        HHL.append(self.ResidualEnthalpy(Tc,Pc,'vapour')+self.EnthalpyIdealGas(Tc,0))
        HHG.append(self.ResidualEnthalpy(Tc,Pc,'vapour')+self.EnthalpyIdealGas(Tc,0)) 
        
        
        for T in Trange:
                H=[]
                prange=scipy.linspace(self.VapourPressure(Tmin,0),Pc,1000)
                for p in prange:
                    H.append(self.RealEnthalpy(T,p))
#                pylab.plot.xlabel('H in J/kmol')
#                pylab.plot.ylabel=('P in Pa')
                pylab.plot(H,prange,'b')
            
        
        pylab.plot(HHL,Psat,'r')   
        pylab.plot(HHG,Psat,'r')
        pylab.show()
        pylab.clf()
         
   
        
        
        

    def PVisotherms(self,Trange):
     '''
       INPUTS
        Trange = [] list of Temperatures in K.
        OUTPUTS
        1. Saturation Curve:
        
        Generates the Saturation P-V envelope for a range of Temperatures in K.
        
        Range = [Tmin,Tmax,1000 data points]
        Tmin = Refernce Temperature in K
        Tmax = Critical Temperature in K
        for some T in Range
        Compute Vapour Pressure Psat in Pa
        Compute <CompressibilityFactors(T,Psat)> for all T in Range
        Convert <CompressibilityFactors(T,Psat)> to volumes.
        
        Plot a graph of Vapour Pressure vs.  Volumes
        
        2. Isotherms
        
        Generates constant temperature P-V curves for all Temperatures in Trange
        
        For a T in Trange 
        
        Compute Vapour Pressure = Psat in Pa
         For a P in the Range of Pressure : 
         Compute <CompressibilityFactors(T,P)[0]> if P<=Psat
         <CompressibilityFactors(T,P)[1])> if P>Psat
         Convert <CompressibilityFactors> to volumes.
         Plot a graph of Range vs. Volume values.
        
        
        Curves are plotted as P vs logV for the sake of clarity.
     '''
            
     Tmax1 = self.CriticalTemperature
     Tmin1 =self.ReferenceTemperature
     TT =scipy.linspace(Tmin1,Tmax1,1000)
     Tc=self.CriticalTemperature
     Pc=self.CriticalPressure
     Psat=[]
     VVL=[]
     VVG=[]
     [Tpmin, Tpmax] = self.VapourPressure('bounds',0)
     [Tvmin, Tvmax] = self.LiquidDensity('bounds',0)
     Tmin = max([Tpmin, Tvmin])
     Tmax = min([Tpmax, Tvmax])
     for T in TT:
            Psat1=float(self.VapourPressure(T,0))
            b=self.CompressibilityFactors(T,Psat1)
            VVL.append(float(b[1])*(self.R)*T/Psat1)
            VVG.append(float(b[0])*(self.R)*T/Psat1)
            Psat.append(float(self.VapourPressure(T,0)))
        
     VVL.append(float(b[0])*(self.R)*Tc/Pc)
     VVG.append(float(b[0])*(self.R)*Tc/Pc)
     Psat.append(Pc)
     
           
        
     
     
     
     
     for T in Trange:
             V=[]
             prange=scipy.linspace(self.VapourPressure(Tmin,0),Pc,1000)
             for p in prange:
              if(p>self.VapourPressure(T,0)):
                V.append(float(self.CompressibilityFactors(T,p)[1])*(self.R)*T/p)
              elif(p<self.VapourPressure(T,0) or T>Tc):
                V.append(float(self.CompressibilityFactors(T,p)[0])*(self.R)*T/p)
              elif p==self.VapourPressure(T,0):
                   list(prange).append(p)
                   V.append(float(self.CompressibilityFactors(T,p)[1])*(self.R)*T/p)
                   V.append(float(self.CompressibilityFactors(T,p)[0])*(self.R)*T/p)
         
             pylab.plot(scipy.log(V),prange,'b')
     


    
     pylab.plot(scipy.log(VVL),Psat,'r')   
     pylab.plot(scipy.log(VVG),Psat,'r')
     pylab.show()
     pylab.clf()
     
    def VapourPressureEOS(self,T):
        '''
        INPUTS: Temperature T in K
        
        OUTPUT: Vapour Pressure at T in Pa.
        
        The Vapour Pressure Psat is given by the equation 
        
        lnphi(Zl,T,Psat)=lnphi(Zg,T,Psat)
        
        Where Zg, Zl = CompressibilityFactors(T,Psat)
        
        Define function funn(P) that computes Zg,Zl = CompressibilityFactors(T,P)
        
        and returns error = exp(lnphi(Zg,T,P))-exp(lnphi(Zl,T,P))
        
        The Vapour Pressure Psat = scipy.optimize.fsolve(funn,guess)
        
        Where guess is suitable guess Pressure in Pa.
        
        '''
        
        
        
        Pguess=self.VapourPressure(273.16,0)
        T1=T
        def funn(P):
            ZG,ZL=self.CompressibilityFactors(T1,P)
            errrr=scipy.exp(self.__getlnphi(float(ZG),T1,P))-scipy.exp(self.__getlnphi(float(ZL),T1,P))
            return errrr
        sol=scipy.optimize.fsolve(funn,Pguess)
        return sol
            

    
    def HvapEOS(self,T):
        '''
        INPUTS : Temperature T in K
        
        OUTPUTS : Enthalpy of Vapourization Hsat in J/kmol
        
        Vapour Pressure Psat is evaluated using the method VapourPressureEOS(T)
        
        Hsat is then given by : 
        
        Hsat = ResidualEnthalpy(T,Psat,'vapour') - ResidualEnthalpy(T,Psat,'liquid')
        
        '''
        
        P=self.VapourPressureEOS(T)
        hv=self.ResidualEnthalpy(T,P,'vapour')
        hl=self.ResidualEnthalpy(T,P,'liquid')
        hh=hv-hl
        return hh
    
    def VlsatEOS(self,T):
        '''
        INPUTS : Temperature T in K
        
        OUTPUTS : Liquid Phase Volume Vlsat in m3/kmol at T and Psat Pa
        
        Vapour Pressure Psat is evaluated using the method VapourPressureEOS(T)
        
        Vlsat is then given by 
        
        Vlsat = Zlsat*R*T/Psat

        
        Where Zgsat, Zlsat = CompressibilityFactors(T,Psat)
        
        '''
        pp=self.VapourPressureEOS(T)
        ZG,ZL=self.CompressibilityFactors(T,pp)
        vll=ZL*self.R*T/pp
        return vll
    
    def adjustparams(self,n,command):
        
        '''
        INPUTS: n  - Number of Data Point
        
        command : either 'plot' or 'none'
        
        This function optimizes the adjustable parameters in the Equations Of State :
        
        1)PRSV1
        2)PRSV2 
        3)Modified PR
        
        Optimization is based on the following Pure Component Properties :
        
        1) Vapour Pressure Psat
        
        2)Enthalpy of Vapourization Hsat
        
        3)Saturated Liquid Phase Volume Vlsat
        
        Range = [Tmin,Tmax,n]
        
        Where Tmin - Lower Bound for correlations for 1), 2) and 3) in K
              Tmax - Upper bound in K
        Define function fgenerror where:
        For T in Range, correlated Psat, Hsat, Vlsat are calculated using the following methods:
        
        1) VapourPressure(T,0) (Pa)
        2)EnthalpyVapourization(T,0) (J/kmol)
        3)LiquidDensity(T,0) (m3/kmol)
        
        The properties are also calculated using equations for state : 
        
        1)VapourPressureEOS(T)
        2)HsatEOS(T)
        3)VlsatEOS(T)
        
        Error calculated as Property(T,0) - PropertyEOS(T,0) for 1), 2) and 3) and stored in list err[]
        
        List of errors is returned.
        The optimized parameters pls are obtained as 
        
        pls = scipy.optimize.leastsq(fgenerror,p0,args)
        
        
        Where p0 is guess value vector for parameters
        
        If command == 'plot' , the graphs of actual values and adjusted values vs T is generated
        
        '''
        if self.EquationOfState=='PRSV2' or self.EquationOfState=='Modified PR' or self.EquationOfState=='PRSV1':
           [Tpmin, Tpmax] = self.VapourPressure('bounds',0)
           [Tvmin, Tvmax] = self.LiquidDensity('bounds',0)
           [Thmin,Thmax]  = self.EnthalpyVapourization('bounds',0)
           Tmin=max(Tpmin,Tvmin,Thmin,273.16)
           Tmax=min(Tpmax,Tvmax,Thmax)
           TT=scipy.linspace(Tmin,Tmax,n)
           Psat=[]
           Hsat=[]
           Vlsat=[]
           Psate=[]
           Hsate=[]
           Vlsate=[]
           if self.EquationOfState=='PRSV2' or self.EquationOfState=='Modified PR':
             p0=[0,0,0]
           else:
             p0=0
           for T in TT:
               pp=self.VapourPressure(T,0)
               a=float(pp)
               Psat.append(a)
               hh=self.EnthalpyVapourization(T,0)
               h=float(hh)
               Hsat.append(h)
               vv=1.0/self.LiquidDensity(T,0)
               Vlsat.append(vv)
               ab=self.VapourPressureEOS(T)
               Psate.append(ab)
               cd=self.HvapEOS(T)
               Hsate.append(cd)
               ef=self.VlsatEOS(T)
               Vlsate.append(ef)
           def fgenerror(p,Psat,Hsat,Vlsat,TT,self):
               if self.EquationOfState=='PRSV2' or self.EquationOfState=='Modified PR':
                   errors=[]
                   self.k1=p[0]
                   self.k2=p[1]
                   self.k3=p[2]
                   self.__setParameters()
               elif self.EquationOfState=='PRSV1':
                   errors=[]
                   self.k1=p
                   self.__setParameters()
               for i in range(0,n):
                   P=self.VapourPressureEOS(TT[i])
                   H=self.HvapEOS(TT[i])
                   V=self.VlsatEOS(TT[i])
                   e1=scipy.fabs((P-Psat[i])/(self.VapourPressure(298.16,0)))
                   e2=scipy.fabs((H-Hsat[i])/(self.EnthalpyVapourization(298.16,0)))
                   e3=scipy.fabs((V-Vlsat[i])*self.LiquidDensity(298.16,0))
                   errors.append(float(e1))
                   errors.append(float(e2))
                   errors.append(float(e3))
               return errors
           pls=scipy.optimize.leastsq(fgenerror,p0,args=(Psat,Hsat,Vlsat,TT,self))
           if command=='plot':
               pylab.plot(TT,scipy.log(Psat),'g')
               pylab.plot(TT,scipy.log(Psate),'r')
               pylab.legend(['Correlation','VP from adjusted '+self.EquationOfState+' EOS'])
               pylab.show()
               pylab.clf()
               pylab.plot(TT,Hsat,'g')
               pylab.plot(TT,Hsate,'r')
               pylab.legend(['Correlation',' Hvap from adjusted '+self.EquationOfState+' EOS'])
               pylab.show()
               pylab.clf()
               pylab.plot(TT,Vlsat,'g')
               pylab.plot(TT,Vlsate,'r')
               pylab.legend(['Correlation','Vlsat from adjusted '+self.EquationOfState+' EOS'])
               pylab.show()
               pylab.clf()
               print pls[0]
           elif command=='none':
               print pls[0]
           else:
               raise ValueError('Incorrect value of command argument. Please mention either plot or none.')
               
    
              
           
           
           
            
        else:
            raise ValueError('wrong EOS')
        
    def plotEOS(self,n):
        '''
        INPUTS : n - Number of Data Points
        
        OUTPUTS : Graphs of 1) Vapour Pressure (Pa) vs T in K
        2) Enthalpy of Vapourization (J/kmol) vs T in K
        
        3) Saturated Liquid Phase Molar Volume (m3/kmol) vs T in K
        
        For T in Trange = [Tmin,Tmax] with n Data Points
        
        The graphs are plotted for 1),2) and 3) for values calculated by correlations and 
        
        those obtained by EquationOfState at all n Temperatures
        
        '''
        [Tpmin, Tpmax] = self.VapourPressure('bounds',0)
        [Tvmin, Tvmax] = self.LiquidDensity('bounds',0)
        [Thmin,Thmax]  = self.EnthalpyVapourization('bounds',0)
        Tmin=max(Tpmin,Tvmin,Thmin,273.16)
        Tmax=min(Tpmax,Tvmax,Thmax)
        TT=scipy.linspace(Tmin,Tmax,n)
        Psat=[]
        Hsat=[]
        Vlsat=[]
        Psate=[]
        Hsate=[]
        Vlsate=[]
        for T in TT:
           pp=self.VapourPressure(T,0)
           a=float(pp)
           Psat.append(a)
           hh=self.EnthalpyVapourization(T,0)
           h=float(hh)
           Hsat.append(h)
           vv=1.0/self.LiquidDensity(T,0)
           Vlsat.append(vv)
           ab=self.VapourPressureEOS(T)
           Psate.append(ab)
           cd=self.HvapEOS(T)
           Hsate.append(cd)
           ef=self.VlsatEOS(T)
           Vlsate.append(ef)
        pylab.plot(TT,scipy.log(Psat),'g')
        pylab.plot(TT,scipy.log(Psate),'r')
        pylab.legend(['Correlation','VP from '+self.EquationOfState+' EOS'])
        pylab.show()
        pylab.clf()
        pylab.plot(TT,Hsat,'g')
        pylab.plot(TT,Hsate,'r')
        pylab.legend(['Correlation',' Hvap from '+self.EquationOfState+' EOS'])
        pylab.show()
        pylab.clf()
        pylab.plot(TT,Vlsat,'g')
        pylab.plot(TT,Vlsate,'r')
        pylab.legend(['Correlation','Vlsat from '+self.EquationOfState+' EOS'])
        pylab.show()
        pylab.clf()
       
class Molecule_UNIQUAC:
    
    
    
        '''
    
    Contains all the methods required to calculate liquid phase activity co-efficients for a mixture of non-electrolytic, non - aggregating molecules using the UNIQUAC activity co-efficient model
    
    The class requires the names of the components forming the Liquid mixture.

    This model is based on statistical mechanical theory, allows local compositions to result from both the size
    and energy differencesbetween the molecules in the mixture. The result is the expression for EXCESS GIBBS ENERGY OF MIXING:

        Gexcess/RT = Gexcess(combinatorial)/RT + Gexcess(residual)/RT

    where the first term accounts for molecular size and shape differences, and the second term accounts largely for energy differences.

        Gexcess(combinatorial)/RT = sum-over-i{ x[i] ln(phi[i]/theta[i] + .5 * z sum-over-j{x[i] * q[i] * ln (theta[i]/phi[i])}}

        Gexcess(residual)/RT = - sum-over-i { q'[i]*x[i]*ln(sum-over-j {theta[j] * tou[j][i]}

        where
                    r[i] = VOLUME PARAMETERS for species i
                    q[i] = SURFACE AREA PARAMETERS for species i
                    q'[i] = SURFACE AREA PARAMETER SUBSTITUTE for molecules forming intermolecular-hydrogen-bonds. for non-hydrogen-bonding molecules, q'[i] = q[i]
                    theta[i] = AREA FRACTION of species i =x[i]*q'[i]/sum-over-j{x[j] * q[j]}
                    phi[i] = VOLUME FRACTION of species i = x[i] * r[i] /sum-over-j{x[j] * r[j]}
                    ln tou[i][j] = -a[i][j]/T
                    where a[i][j] is the BINARY INTERACTION PARAMETER between i th and j th species
                    z = co-ordination number = 10
        Formulae for calculation of the ACTIVITY COEFFICIENT gamma[i]:
            ln(gamma[i]) = ln(gamma(combinatorial)[i])+ln(gamma(residual)[i])
        where,
            ln(gamma(combinatorial)[i]) = ln( phi[i]/x[i]) - z*0.5 *q[i]*ln(phi[i]/theta[i]) + l[i] - phi[i]*sum-over-j{x[j]*l[j]}/x[i]
            where,
                l[i] = (r[i]-q[i])*.5*z - r[i] +1
            ln(gamma(residual)[i]) = q'[i] (1 - ln(sum-over-j{theta[j]*tou[j][i]} - sum-over-j{ theta[j]*tou[i][j]/sum-over-k{ theta[k] * tou[k][j]}}
  ******************************************************************************
  The following are attributes of Molecule_UNIQUAC
  
  ******************************************************************************
  listr : It is a dictionary containing the volume parameters of each species 
          in the mixture
  listq : It is a dictionary that contains the surface area parameters of the 
          mixture components
  listqd: Is also a dictionary containing the adjusted surface area parameter values
  
  listas: Is a dictionary containing the interaction parameters for each binary pair
          in the mixture.
  
  ***************************************************************************
    The following private utility methods are available
    ***************************************************************************
    __init__                    : Defines the attributes reuired for the UNIQUAC method and assigns values to them by calling __readparameterfile() and __readrand()
    __readparameterfile         : Uses the 'interactionparameters.txt' text file as an input to obtain the BINARY INTERACTION PARAMETERS for all the components comprising the vapour-liquid system
    __readrandq                 : Uses the 'rqparameters.txt' text file as an input to obtain the volume (r) and surface area (q) parameters for each molecular
                                  species for use in the UNIQUAC model
    ***************************************************************************
    The following public methods are available
    ***************************************************************************
    createlists     : INPUT = Liquid phase composition of the mixture as a list ; RETURNS it as a dictionary after checking whether the sum is one and that all the molefractions are positive  
    getTau          : INPUT = Temperature ; RETURNS a dictionary of tou[i][j], the binary parameters calculated using self.listas of binary interaction parameters or the interaction parameters' list
    getPhi          : INPUT = Compostion of the liquid mixture as a list; CALLS createlists to convert it into a dictionary; CALCULATES phi values for all components and stores them in class attribute self.listphi as a dictionary and RETURNS its copy aaa 
    getTheta        : INPUT = Compostion of the liquid mixture as a list; CALLS createlists to convert it into a dictionary; CALCULATES theta values for all components and stores them in class attribute self.listtheta as a dictionary and RETURNS its copy aaa 
    getGexComb      : INPUT = Temperature, composition; Calls gettheta, gettau @ Temperature and getphi to obtain values of theta,tau and phi, CALLS createlists to get a dictionary of compostions which is used along with the attributes of Molecule_UNIQUAC like r,q to calculate the GexcessCombinatorial which is then RETURNED
    getGexRes       : INPUT = Temperature, composition; Calls gettheta, gettau @ Temperature and getphi to obtain values of theta,tau and phi, CALLS createlists to get a dictionary of compostions which is used along with the attributes of Molecule_UNIQUAC like r,q to calculate the GexcessResidual which is then RETURNED
    getGex          : INPUT = Temperature and compostion; CALLS getgexcomb and getgexres by passing Temperature and Composition as the attributes and RETURNS Total GIBBSexcess for the multicomponent system
    getGammaComb    : INPUT = Temperature and composition; CALLS gettau @ Temperature,getphi and getheta and uses the formulae given above to obtain and RETURNS values of GammaCombinatorial as a Dictionary
    getGammaRes     : INPUT = Temperature and composition; CALLS gettau @ Temperature,getphi and getheta and uses the formulae given above to obtain and RETURNS values of GammaResidual as a Dictionary
    getGamma        : INPUT = Temperature, composition; CALLS getgammacomb and getgammares to obtain their respective values and multiplies them to obtain and return overall gamma for each of the components as a dictonary
    getGammaInfDil  : This function is not needed as far as the UNIQUAC model is concerned but however, for doing the VLE calculations using Wong-Sandler method, the value of gamma@infinite dilution for all the components forming the mixture.
                      This function assigns a molefraction of 1e-10 to  the component whose gamma@infi dilution is to be obtained and distributes the remaining fraction (1 - 1e-01) amongst other components. Then it creates a new molefraction list which is passed to the method getgamma to obtain Activity Coefficient at infintite dilution for a given species.
        '''
        def __init__ (self,componentlist):
            '''
                    The class intanciation:: obj = Molecule_UNIQUAC(name) ::requires the list of names of  components of the mixture to be specified.
                    
                    This will direct the code to read the datafiles called interactionparameters.txt and rqparameters.txt and use the data to ready the object for use.
            
            INPUT : list of components
            
            CALLS : __readparameterfile()
                    __readrandq()                
            '''
           
            self.componentlist=copy.copy(componentlist)
#            self.objectslist='junk'
            self.listas='junk'
            self.listr='junk'
            self.listq='junk'
            self.listqd='junk'
            self.R=8314.0 #J/KmolK
            self.listtaucomp=[]
            for i in self.componentlist:
                    for j in self.componentlist:
                        if i!=j:
                            self.listtaucomp.append([i,j])
            self.__readparameterfile() 
            self.__readrandq()
            
        def checklist(self,complist):
            '''
            INPUTS 
            
            complist: A dictionary containing the composition of the mixture in mole fraction form.
            
            OUTPUT: 
            
            The checklist method is used to check if the values of mole fraction associated with the given species in mixture are physically meaningful. This 
            method checks the input mole fractions complist for the following:
            
            1) No element must be zero or negative
            
            2) The sum of the elements must be equal to 1
            
            The method is designed to crash the entire program if either of the above requirements is not met.
            
            '''
            sum=0.0
            for i in dict.keys(complist):
                    if complist[i]<=0:
                        raise ValueError('atleast one mole fraction is zero or negative. Please use Molecule_CubicEquationOfState class for pure objects')
            for i in dict.keys(complist):    
                    sum=sum+complist[i]
            if scipy.fabs(sum-1)>1e-10:
                        raise ValueError('mole fractions don\'t add up to 1')
        
        def createlists(self,compositionlist,mode):
            '''
             INPUTS  
             compositionlist : that is a list of the liquid phase composition (molefraction or x) of the mixture 
             
             mode: Can take either '0' or '1'. If mode==0, the checklist() method is executed on the dictionary. If mode==1, no check is 
             
             performed. If mode takes any other value, a ValueError is raised. 
             
             RETURN = dictcomps : dictionary of x after checking whether summation of all x =1 and all x are positive depending upon the value of mode
            '''
            
            
            n=len(self.componentlist)
            n1=len(compositionlist)
            

            if n==n1:
                dictcomps={}
                for a in self.componentlist:
                    yyy=self.componentlist.index(a)
                    dictcomps[a]=compositionlist[yyy]
                sum=0.0
                
                if mode==0:
                    self.checklist(dictcomps) 
                elif mode!=1:
                   raise ValueError('invalid numerical parameter. Please enter 0 or 1')
            else:
                raise ValueError('Wrong format of input')
            return dictcomps
        
       
            
            
        def __readparameterfile(self):
            '''
             OPEN = interactionparameters.txt
             
             OBTAIN = binary interaction parameters (listas) as a dictionary for all the pairs possible from the given liquid mixture

             
            '''
            

            filename1 = 'interactionparameters.txt'
            if os.path.isfile(filename1):
                f = open(filename1,'r')
                data = f.readlines()
                f.close()            
                dputname = data[0].split()[0]  #The name is the very first bit of text in the file
                if dputname != 'BinaryInteractionParameters':
                    print 'There is something wrong in '+filename1+'. Does not match with '+dputname+'.'
                    raise NameError()
                else:
                    self.listas={}
                    com=copy.copy(self.listtaucomp)
                    for idata in range(len(data)):
                        if idata > 0:
                            info = data[idata]    
                            info = info.split()
                            name1 = info[0]
                            name2 = info[1]
                            for i in self.componentlist:
                                for j in self.componentlist:
                                    if i==j:
                                        self.listas[i,j]=0.0
                                    elif i==name1 and j==name2:
                                            try:
                                                value=float(info[2])
                                                self.listas[i,j]=value
                                                junk=com.pop(com.index([i,j]))
                                            except ValueError:
                                                print 'ERROR in reading parameters for '+i+' and '+j+' pair'
                    
                    if len(com) > 0:
                        print 'The parameters were not available in '+filename1+' for the following pairs:'
                        for i in range(len(com)) :
                            print ''+com[i][0]+' - '+com[i][1]+''
            else:
                print ''+filename1+' does not exist'
         
            
         
    

    
            


        def __readrandq(self):
            
            '''
             OPEN : rqparameters.txt which supplies the values of volume and area interaction parameters for various compounds

             OBTAIN : Store the values of Volume and Surface Area Parameters in 3 different dictionaries which
             
             are also attributes of the class ( listr for r, listq for q, listqd for q')
            '''
            
            filename1 = 'rqparameters.txt'
            if os.path.isfile(filename1):
                f = open(filename1,'r')
                data = f.readlines()
                f.close()            
                dputname = data[0].split()[0]  #The name is the very first bit of text in the file
                if dputname != 'rqParameters':
                    print 'There is something wrong in '+filename1+'. Does not match with '+dputname+'.'
                    raise NameError()
                else:
                    self.listr={}
                    self.listq={}
                    self.listqd={}
                    comp=copy.copy(self.componentlist)
                    comp1=copy.copy(self.componentlist)
                    for idata in range(len(data)):
                        if idata > 0:
                            name = data[idata].split()[0]
                            for a in self.componentlist:
                                if a==name:
                                    info1 = data[idata+1]
                                    info2 = data[idata+2]
                                    info3 = data[idata+3]
                                    info1=info1.split()
                                    info2=info2.split()
                                    info3=info3.split()
                                    if info1[0] == 'r' and info2[0]=='q':
                                        try:
                                            valuer = float(info1[1])
                                            self.listr[a]=valuer
                                            aaa=comp.pop(comp.index(a))
                                            valueq = float(info2[1])
                                            self.listq[a]=valueq
                                            bbb=comp1.pop(comp1.index(a))
                                            if info3[0]=='q-':
                                                self.listqd[a]=float(info3[1])
                                            else:
                                                self.listqd[a]=valueq
                                                
                                        except ValueError:
                                            print 'ERROR in reading r and q parameters file. Check file data.'
                    if len(comp)>0:
                        print 'The r parameters were not available in '+filename1+' for the following compounds:'
                        for a in comp:
                            print a
                    if len(comp1)>0:
                        print 'The q parameters were not available in '+filename1+' for the following pairs:'
                        for a in comp1:
                            print a
        
            else:
                raise NameError(''+filename1+' does not exist')
        
        def getTau(self,T):
            
            '''
             INPUT : T ; Temperature in Kelvin
             
             RETURN : taus ; Dictionary consisting of tou values for various pairs of components in the mixture by using their listas i.e. interaction parameter values
            '''
            taus={}
            for a in self.componentlist:
                for b in self.componentlist:
                    taus[a,b]=scipy.exp(-self.listas[a,b]/T)
            return taus
            
        def getThetad(self,comp,ll):
            
            '''
             INPUT : list comp ; composition in mole fraction
                     ll(mode): Either '0' or '1'
             
             CALL : creatlists(comp,ll) to convert list composition to dictionary composition
             
             ASSIGN : theta(area fraction) values to class attribute listtheta dictionary
             
             RETURN : listqd (adjusted area fraction) dictionary 
            '''
            xxx=self.createlists(comp,ll)
            qd={}
            self.listthetad={}
            sum=0
            for a in self.componentlist:
                sum=sum+xxx[a]*self.listqd[a]
            for a in self.componentlist:
                qd[a]=xxx[a]*self.listqd[a]/sum
            return qd
            
            
            
        def getPhi(self,comp,ll):
            '''
             INPUT : list comp ; composition in mole fraction
                     ll (mode) : Either '0' or '1' 
             
             CALL : creatlists(comp,ll) to convert list composition to dictionary composition
             
             ASSIGN : phi(volume fraction) values to class attribute listphi dictionary
             
             RETURN : listphi (volume fraction) dictionary
            '''
            
            
            xxx=self.createlists(comp,ll)
            listphi1={}
            self.listphi={}
            sum=0
            for a in self.componentlist:
                sum=sum+xxx[a]*self.listr[a]
            for a in self.componentlist:
                listphi1[a]=xxx[a]*self.listr[a]/sum
                self.listphi=copy.copy(listphi1)
            return listphi1
        def getTheta(self,comp,ll):
            '''
             INPUT : list comp ; composition in mole fraction
                     ll(mode): Either '0' or '1'
             
             CALL : creatlists(comp,ll) to convert list composition to dictionary composition
             
             ASSIGN : theta(area fraction) values to class attribute listtheta dictionary
             
             RETURN : thet (area fraction) dictionary 
            '''
            
            xxx=self.createlists(comp,ll)
            thet={}
            self.listtheta={}
            sum=0
            for a in self.componentlist:
                sum=sum+xxx[a]*self.listq[a]
            for a in self.componentlist:
                thet[a]=xxx[a]*self.listq[a]/sum
            return thet
        def getGexComb(self,T,compp,ll):
            '''
            INPUT : list compp : composition in mole fraction
                    
                    float T : Temperature in K
                    
                    ll(mode): Either '0' or '1'

            CALL : gettheta(compp) to obtain theta
                   
                   getphi(compp) to obtain phi
                   
                   createlists to convert list(comp) composition to dictionary composition
                   
                   gettau(T) to obtain tou for all binary-pairs in the system
            
            RETURN : float gc, Gibbs Excess Combinatorial in (J/kmol)
            '''
            
            R=self.R
            sum1=0.0
            sum2=0.0
            theta=self.getTheta(compp,ll)
            phi=self.getPhi(compp,ll)
            x=self.createlists(compp,ll)
            r=copy.copy(self.listr)
            q=copy.copy(self.listq)
            tau=self.getTau(T)
            for a in self.componentlist:
                sum1=sum1+x[a]*scipy.log(phi[a]/x[a])
                sum2=sum2+x[a]*q[a]*scipy.log(theta[a]/phi[a])
            ggbyrt=sum1+5.0*sum2
            gc=R*T*(ggbyrt)
            return gc
        def getGexRes(self,T,compp,ll):
            '''
            INPUT : list compp ; composition in mole fraction
                    
                    float T ; Temperature in K
                    
                    ll(mode): Either '0' or '1'
                    
            CALL : gettheta(compp) to obtain theta
                   
                   getphi(compp) to obtain phi
                   
                   gettau(T) to obtain tou for all binary-pairs in the system
                   
                   createlists to convert list(comp) composition to dictionary composition
            
            RETURN : float gr ; Gibbs Excess Residual in (J/kmol)
            '''
            
            R=self.R
            sum2=0.0
            theta=self.getTheta(compp,ll)
            thetad=self.getThetad(compp,ll)
            phi=self.getPhi(compp,ll)
            x=self.createlists(compp,ll)
            r=copy.copy(self.listr)
            q=copy.copy(self.listq)
            qd=copy.copy(self.listqd)
            tau=self.getTau(T)
            for a in self.componentlist:
                sum1=0.0
                for b in self.componentlist:
                    sum1=sum1+thetad[b]*tau[b,a]
                sum2=sum2+qd[a]*x[a]*scipy.log(sum1)
            gr=-1.0*sum2*R*T
            return gr
                    
            
            
            
        def getGex(self,T,compp,ll):
            '''
             INPUT : float T ; Temperature in Kelvin
                     
                     list compp ; liquid phase composition in molefraction
                     
                     ll(mode): Either '0' or '1'
             CALL : getexcomb(T,compp) and getexres(T,compp) to get the values of Gibbs Excess Combinatorial and Gibbs Excess Residual respectively at T and compp
             
             RETURN : float ge ; Gibbs Excess at T and compp in J/kmol
            '''
            ge=self.getGexComb(T,compp,ll)+self.getGexRes(T,compp,ll)
            return ge
        def getGammaComb(self,T,compp,ll):
            '''
             INPUT : float T ; Temperature in Kelvin
                     
                     list compp ; liquid phase composition in molefraction
                     
                     ll(Mode) : Either '0' or '1'
             
             CALL : gettheta(compp) to obtain theta
                    
                    getphi(compp) to obtain phi
                    
                    gettau(T) to obtain tou for all binary-pairs in the system
                    
                    createlists to convert list(comp) composition to dictionary composition
            
            RETURN : dictionary gammac ; Combinatorial Gamma
            '''
            
            R=self.R
            sum1=0.0
            sum2=0.0
            sum3=0.0
            theta=self.getTheta(compp,ll)
            phi=self.getPhi(compp,ll)
            x=self.createlists(compp,ll)
            r=copy.copy(self.listr)
            q=copy.copy(self.listq)
            tau=self.getTau(T)
            lngammac={}
            gammac={}
            lngammar={}
            l={}
            
            for a in self.componentlist:
                l[a]=(r[a]-q[a])*(5.0)-(r[a]-1.0)
            for a in self.componentlist:
                sum3=sum3+l[a]*x[a]
            for a in self.componentlist:
                lngammac[a]=scipy.log(phi[a]/x[a])-(10.0/2.0)*q[a]*scipy.log(phi[a]/theta[a])+l[a]-(phi[a]/x[a])*sum3
                gammac[a]=scipy.exp(lngammac[a])
            return gammac
            
        def getGammaRes(self,T,compp,ll):
            '''
             INPUT : float T ; Temperature in Kelvin
                     
                     list compp ; liquid phase composition in molefraction
                     
                     ll(mode) : Either '0' or '1'
             
             CALL : gettheta(compp) to obtain theta
                    getphi(compp) to obtain phi
                    gettau(T) to obtain tou for all binary-pairs in the system at T
                    createlists to convert list(comp) composition to dictionary composition
             RETURN : dictionary gammar ; Residual Gamma
            '''
            
            lngammar={}
            gammar={}
            R=self.R
            thetad=self.getThetad(compp,ll)
            theta=self.getTheta(compp,ll)
            phi=self.getPhi(compp,ll)
            x=self.createlists(compp,ll)
            r=copy.copy(self.listr)
            q=copy.copy(self.listq)
            qd=copy.copy(self.listqd)
            tau=self.getTau(T)
            for a in self.componentlist:
                sum1=0.0
                sum3=0.0
                for b in self.componentlist:
                    sum1=sum1+thetad[b]*tau[b,a]
                    sum2=0.0
                    for c in self.componentlist:
                        sum2=sum2+thetad[c]*tau[c,b]
                    sum3=sum3+thetad[b]*tau[a,b]/sum2
                lngammar[a]=qd[a]*(1-scipy.log(sum1)-sum3)
                gammar[a]=scipy.exp(lngammar[a])
            return gammar
        def getGamma(self,T,compp,ll):
            '''
             INPUT : float T ; Temperature in Kelvin
                     list compp ; liquid phase composition in molefraction
             
             CALL : getgammacomb(T,compp) and getgammares(T,compp) to get the values of COMBINATORIAL GAMMA AND RESIDUAL GAMMA 
             
             RETURN : dictionary gama ; Gamma
            '''
            
            gama={}
            for a in self.componentlist:
                gama[a]=self.getGammaComb(T,compp,ll)[a]*self.getGammaRes(T,compp,ll)[a]
            return gama
        def getGammaInfDil(self,T,name,ll):
            '''
             INPUT : float T ; Temperature in Kelvin
                     
                     string name ; name of the component whose gamma@infinite dilution is to be found
             
                     ll(mode): Either '0' or '1'
             
             CALL : getgamma(T, artificial-molefraction) to get gamma@infinite dilution
             
             RETURN : float ginfd ; Gamma@infinite Dilution
            '''
            dd=1e-10
            xxdef=[]
            n=len(self.componentlist)
            for i in range(len(self.componentlist)):
                if self.componentlist[i]==name:
                    xxd=dd
                else:
                    xxd=(1.0-dd)/(n-1.0)
                xxdef.append(xxd)
            ginfd=self.getGamma(T,xxdef,ll)[name]
            return ginfd
                
            
            
            

class WongSandlerVLE(Molecule_UNIQUAC):
    '''
    This class contains all the attributes and methods required to perform the Equation of State based Vapour Liquid Equilibrium calculation for a mixture of non - electrolytic, non - aggregative molecules.
    
    The 'WongSandlerVLE' class inherits all the parameters and functions from the 'Molecule_UNIQUAC' parent class, whose methods are used in evaluating liquid phase interaction parameters; Other activity co-efficient models and their modules will be incorporated to aid accurate calculations.
    The class requires the names of the molecules in the stream, the equation of state name: either 'PRSV1'or 'PRSV2' or 'Modified PR' for Peng-Robinson.

    ***************************************************************************
        The following are the SCALAR attributes of this class:
    ***************************************************************************  
    yguess		: The dictionary list of the guess values of the vapor phase mole fraction of the compounds given
    pguess		: The guess value of pressure for calculations
    objects		: The list of instances of class Molecule_CubicEquationOfState which enable us to obtain pure fluid thermodynamic properties.
    eosname		: The name of the Equation of State being used
    listk		: The list of the k values to be used in Wong Sandler method
    T		: The temperature input at which the calculations are to be carried out given by the user.


    ***************************************************************************
        The following private utility methods are available
    ***************************************************************************
    __createobjectslist 	: 	creates a dictionary list of compounds in the stream alongwith their properties by calling adjustparams from class Molecule_CubicEquationOfState
    __guesstimates	        :	Requires obj as the list of instances of class Molecule_CubicEquationOfState 
                                         comp as list of liquis phase compositions in the inlet
    			                Assigns the guess value of Pressure as the sum of the products of mole fraction of components and their vapour pressures at the given temperature
    			                Uses function getfugacoeff() of Molecule_CubicEquationOfState to calculate the fugacity coefficients in the vapor and the liquid phase which is then used to caluclate the guess values of 			the vapor phase mole fraction as per:
                                yi*phiv = xi*phil				
			

     ***************************************************************************
        The following public methods are available
     ***************************************************************************
    createk		:	Creates a list of k values 'listk' for the compounds in the componentlist as per: If p and q are two species in the stream then:
    				kp = 1 - [(bq-aq/(R*T))*((gamainp+lphip-Mp)/(Zq-1))+bq*(1+gamainp/c-ap/(bp*R*T))]/[bp-ap/(R*T)+bq-aq/(R*T)]
    				kq = 1 - [(bp-ap/(R*T))*((gamainq+lphiq-Mq)/(Zp-1))+bp*(1+gamainq/c-aq/(bq*R*T))]/[bp-ap/(R*T)+bq-aq/(R*T)]			
    			where  	Mp = -log(((Vq-bq)*Zq)/Vq)+(1.0/(e-s))*((ap/(bp*R*T))-(gamainp)/c)*(scipy.log((Vq+s*bq)/(Vq+e*bq)))	
    			        Mq = -log(((Vp-bp)*Zp)/Vp)+(1.0/(e-s))*((aq/(bq*R*T))-(gamainq)/c)*(scipy.log((Vp+s*bp)/(Vp+e*bp)))	
    			        s = sigma 
    				    e = epsilon
    
    getkpq		: 	Creates a list of kpq values 'listk1' based on the k values in listk for all the compounds present in the stream(to calculate Epq):
    				kpq = kp*xq + kq*xp
    			
    
    getepq		:	Creates a list of Epq values in 'listepq' for for the compounds in the componentlist (to calculate the parameter D) as per:
    				Epq = 0.5*(bp-ap/(R*T)+bq-aq/(R*T))*(1-kpq)
    
    getD		:	Gets the value of D to be used for calculation of parameters a and b:
    				D=1+(Gexcess/(c*R*T))-sum (xp*(ap/(bp*R*T))
    
    mixingrules	:	Gets the values of mixing parameters a and b for the given mixture:
    				bmix = (1/D)*sum(xp*xq*Epq)
    				amix = bmix*R*T*(1-D)
    
    bbars		:	Creates a list of the values of partial parameter bbar for each component in the stream:
                    For any two species i and j:
    				bbar = (1/D)*(2*sum(xj*Eij)-bmix*(1+(ln(gamma(i))/c)-a(i)/(b(i)*R*T)))
    
    abars		:	Creates a list of the values of partial parameter abar for each component in the stream:
                    For any two species i and j:
    				abar = bmix*R*T*(ai/(bi*R*T)-ln(gamma(i))/c)+amix*(bi/bmix-1)	
                    
    mixturevolume	:	Returns the volume of the mixture V after caluclating the compressibility factor Z :
                        The equation of state can be cast in the form of a cubic equation in the compressibility Z = PV/RT
                                			Z**3 + A*Z**2 + B*Z + C = 0
                        where,	pp   = P*b/(R*T)
           				        ppp  = (1.0 + pp)/pp
            				    ap   = a/(P*b**2)
            				    A =  pp*(epsilon + sigma - ppp)
                                B =  pp**2*(ap + sigma*epsilon - (epsilon + sigma)*ppp)
            				    C = -pp**3*(ap + sigma*epsilon*ppp) 
    
    phibars		:	Returns the list of activity coeffient phibar for each component which is calculated as:
    				xx=(amix/(b*R*T))/(epsilon-sigma)
                    yy=1+abar/amix-bbar/bmix
               		zz=log((V+sigma*bmix)/(V+epsilon*bmix))
               		lnphibars=(bbar/bmix)*(Z-1)-log((V-bmix)*Z/V)+xx*yy*zz
               		phibars=exp(lnphibars)
    
    doVLEcalc	:	Requires inlet liquid phase mole fractions as a list
    			    Returns the vapor phase mole fractions and the equilibrium pressure in Pascals
    
    doflashcalc	:	Requires inlet composition of compounds and the final pressure as input variables
    			    Returns the fraction of feed vapourized, amount in liquid phase and amount in vapour
    '''
    def __init__(self,componentlist,eosname,T):
        '''
        INPUT : componentlist : list of components in the stream
                eosname       : the equation of state name: either 'PRSV1'or 'PRSV2' or 'Modified PR' for Peng-Robinson
                T             : the temperature at which the calculations are to be carried out
        CALLS:  createk     : to calculate values of k
        '''
        Molecule_UNIQUAC.__init__(self,componentlist)
        self.yguess='junk'
        self.pguess='junk'
        self.eosname=eosname
        self.objects='junk'
        self.listk='junk'
        self.T=T
        self.createk()
      
        
    def __createobjectslist(self):
        '''
        creates a dictionary list of compounds in the stream alongwith their properties 
        CALLS :   adjustparams from class Molecule_CubicEquationOfState
        '''
        self.objects={}
        for a in self.componentlist:
            self.objects[a]=Molecule_CubicEquationOfState(a,self.eosname)
            if self.eosname=='PRSV1' or self.eosname=='PRSV2' or self.eosname=='Modified PR':
                self.objects[a].adjustparams(5,'none')
    def __guesstimates(self,obj,comp):
        '''
        INPUT: obj                      :list of compounds in the stream as objects of class Molecule
               comp                     :liquid phase compositions of each compound in the stream
        CALLS: VapourPressure       :calculates the vapour pressure of the component at the given temperature
               CompressibilityFactors :calculates the compressibility factors of the compound at the given temperature and pressure
               getfugacoeff           :calculates the fugacity coefficients at the given temperature and pressure
        '''
        listx=self.createlists(comp,0)
        T=self.T
        self.yguess={}
        sum=0
        sum1=0
        for a in self.componentlist:
            sum+=obj[a].VapourPressure(T,0)*listx[a]
        self.pguess=sum
        p=sum

        
        for a in self.componentlist:
            aa=obj[a].CompressibilityFactors(T,p)
            phiv=obj[a].getfugacoeff(float(aa[0]),T,p)
            phil=obj[a].getfugacoeff(float(aa[1]),T,p)
            self.yguess[a]=float(listx[a]*phil/phiv)
            sum1=sum1+self.yguess[a]
        for a in self.componentlist:
            self.yguess[a]=self.yguess[a]/sum1
    
            
    def createk(self):
        '''
        CALLS : VapourPressure:calculates the vapour pressure of the component at the given temperature through its respective class instance
                
                CompressibilityFactors :calculates the compressibility factors of the compound at the given temperature and pressure using its class instance
                
                getfugacoeff:calculates the fugacity coefficients at the given temperature and pressure with its class instance.
                
                getGammaInfDil  :from Molecule_UNIQUAC to calculate the activity coefficient at infinite dilution with its class instance.
        '''
        self. __createobjectslist()
        T=self.T
        self.listk={}
        obj=copy.copy(self.objects)
        R=self.R
        
        for p in self.componentlist:
            for q in self.componentlist:
                ppp=obj[p].VapourPressure(T,0)
                ppq=obj[q].VapourPressure(T,0)
                Zp=float(obj[p].CompressibilityFactors(T,ppp)[1])
                Zq=float(obj[q].CompressibilityFactors(T,ppq)[1])
                Vp=Zp*R*T/ppp
                Vq=Zq*R*T/ppq
                s=obj[p].sigma
                e=obj[p].epsilon
                ap=obj[p].a(T,0)
                aq=obj[q].a(T,0)
                bp=obj[p].b
                bq=obj[q].b
                Zp1=float(obj[p].CompressibilityFactors(T,ppq)[1])
                Zq1=float(obj[q].CompressibilityFactors(T,ppp)[1])
                lphip=scipy.log(obj[p].getfugacoeff(Zp1,T,ppq))
                lphiq=scipy.log(obj[q].getfugacoeff(Zq1,T,ppp))
                if p==q:
                    gamainp=1.0
                    gamainq=1.0
                else:
                    bbcd=Molecule_UNIQUAC([p,q])
                    gamainp=scipy.log(bbcd.getGammaInfDil(T,p,0))
                    gamainq=scipy.log(bbcd.getGammaInfDil(T,q,0))
                c=obj[p].c
                Mp=-scipy.log(((Vq-bq)*Zq)/Vq)+(1.0/(e-s))*((ap/(bp*R*T))-(gamainp)/c)*(scipy.log((Vq+s*bq)/(Vq+e*bq)))
                Mq=-scipy.log(((Vp-bp)*Zp)/Vp)+(1.0/(e-s))*((aq/(bq*R*T))-(gamainq)/c)*(scipy.log((Vp+s*bp)/(Vp+e*bp)))
                dp=(bq-aq/(R*T))*((gamainp+lphip-Mp)/(Zq-1))+bq*(1+gamainp/c-ap/(bp*R*T))
                dq=(bp-ap/(R*T))*((gamainq+lphiq-Mq)/(Zp-1))+bp*(1+gamainq/c-aq/(bq*R*T))
                ep=bp-ap/(R*T)+bq-aq/(R*T)
                kp=float(1-dp/ep)
                kq=float(1-dq/ep)
                self.listk[p,q]=[kp,kq]

         
    def get_kpq(self,compp,ll):
        '''
        INPUT   : compp       : List of mole fractions of each species in mixture.
                  ll          : Mode, may take '0' or '1'
        CALLS   : createlists : Converts the list of compositions to a dictionary form.
        RETURNS : list of kpq values for the compounds in the stream
        '''
        x=copy.copy(compp)
        listk1={}
        comp=self.createlists(compp,ll)
        for a in self.componentlist:
            for b in self.componentlist:
                listk1[a,b]=float(comp[b]*self.listk[a,b][0]+comp[a]*self.listk[a,b][1])
        return listk1
    def get_epq(self,comp,ll):
        '''
        INPUT   : compp       : List of mole fractions of each species in mixture.
                  ll          : Mode, may take '0' or '1'
        CALLS   : get_kpq     :creates a list of kpq values for compounds in the stream
        RETURNS : list of Epq values for the compounds in the stream
        '''
        R=self.R
        T=self.T

        listepq={}
        obj=copy.copy(self.objects)
        listkk=self.get_kpq(comp,ll)
        for a in self.componentlist:
            for b in self.componentlist:
                ap=obj[a].a(T,0)
                aq=obj[b].a(T,0)
                bp=obj[a].b
                bq=obj[b].b
                listepq[a,b]=float(0.5*(bp-ap/(R*T)+bq-aq/(R*T))*(1-listkk[a,b]))
        return listepq
    def getD(self,compp,ll):
        '''
        INPUT   : compp       : List of mole fractions of each species in mixture.
                  ll          : Mode, may take '0' or '1'
        CALLS   : get_kpq    :creates a list of kpq values for compounds in the stream
                  get_epq    :creates a list of Epq values for compounds in the stream
                  createlists:
                  getGex    : Returns the excess Gibb's Energy of mixing for the mixture at the given Temperature in J/kmol
        RETURNS : D value for the compounds in the stream
        '''
        T=self.T

        listk1=self.get_kpq(compp,ll)
        listepq=self.get_epq(compp,ll)
        comp=self.createlists(compp,ll)
        R=self.R
        s=0.0
        obj=copy.copy(self.objects)
        for a in self.componentlist:
            s=s+(comp[a]*(obj[a].a(T,0))/(obj[a].b*R*T))
            c=obj[a].c
        D=1+(self.getGex(T,compp,ll)/(c*R*T))-s
        return D
    def mixingrules(self,compp,ll):
       '''
        INPUT   : compp       : List of mole fractions of each species in mixture.
                  ll          : Mode, may take '0' or '1'
        CALLS   : get_kpq    :creates a list of kpq values for compounds in the stream
                  get_epq    :creates a list of Epq values for compounds in the stream
                  createlists:
                  getD     :gives the D value for the compounds in the stream
        RETURNS : mixing parameters amix, bmix for the compounds in the stream
       '''
       T=self.T

       kpq=self.get_kpq(compp,ll)
       R=self.R
       epq=self.get_epq(compp,ll)
       comp=self.createlists(compp,ll)
       D=self.getD(compp,ll)
       s=0.0
       for a in self.componentlist:
           for b in self.componentlist:
               s=s+(1/D)*(comp[a]*comp[b]*epq[a,b])
       bmix=float(s)
       amix=float(bmix*R*T*(1-D))
       return [amix,bmix]
    def bbars(self,comp,ll):
        '''
        INPUT   : compp       : List of mole fractions of each species in mixture.
                  ll          : Mode, may take '0' or '1'
        CALLS   : getD     :gives the D value for the compounds in the stream
                  get_epq    :creates a list of Epq values for compounds in the stream
                  mixingrules:gives the amix, bmix values for the compounds in the stream
                  createlists: Converts the list of compositions to a dictionary form. 
                  getGamma   : returns the activity co-efficients dictionary for the mixture
        RETURNS : bbars        :list of partial parameter bbar for the compounds in the stream
        '''
        T=self.T

        D=self.getD(comp,ll)
        R=self.R
        epq=self.get_epq(comp,ll)
        [amix,bmix]=self.mixingrules(comp,ll)
        amix=float(amix)
        bmix=float(bmix)
        compp=self.createlists(comp,ll)
        o=copy.copy(self.objects)
        bbars={}
        for a in self.componentlist:
            s=0.0
            lny=scipy.log(self.getGamma(T,comp,ll)[a])
            for b in self.componentlist:
                s=s+compp[b]*epq[a,b]
            bbars[a]=float((1/D)*(2*s-bmix*(1+(lny/o[a].c)-o[a].a(T,0)/(o[a].b*R*T))))
        return bbars
    def abars(self,comp,ll):
        '''
        INPUT   : compp       : List of mole fractions of each species in mixture.
                  ll          : Mode, may take '0' or '1'
        CALLS   : getD       :gives the D value for the compounds in the stream
                  get_epq    :creates a list of Epq values for compounds in the stream
                  mixingrules:gives the amix, bmix values for the compounds in the stream
                  createlists:Converts the list of compositions to a dictionary form.
                  bbars     : returns list of partial parameter bbar for the compounds in the stream
                  getGamma   : returns the list of activity coefficients for each component in the mixture
        RETURNS : abars        :list of partial parameter abar for the compounds in the stream
        '''
        T=self.T

        D=self.getD(comp,ll)
        R=self.R
        epq=self.get_epq(comp,ll)
        [amix,bmix]=self.mixingrules(comp,ll)
        amix=float(amix)
        bmix=float(bmix)
        compp=self.createlists(comp,ll)
        o=copy.copy(self.objects)
        bbars=self.bbars(comp,ll)
        abars={}
        for a in self.componentlist:
            ai=o[a].a(T,0)
            bi=o[a].b
            c=o[a].c
            lnyi=scipy.log(self.getGamma(T,comp,ll)[a])
            abars[a]=float(bmix*R*T*(ai/(bi*R*T)-lnyi/c)+amix*(bbars[a]/bmix-1))
        return abars
    
                
    def mixturevolume(self,comp,state,P,ll):
        '''
        INPUT   : 
                  comp      : List of mole fractions of each species in mixture.
                  ll          : Mode, may take '0' or '1'
                  state        : Either 'liquid' or 'vapour'
                  P            : the given pressure in Pa 
        CALLS   : mixingrules:gives the amix, bmix values for the compounds in the stream
                  Utilities.cubicsolver: gives the solution to a cubic equation on passing the coefficients A,B,C
        RETURNS : V            : the volume of the mixture 
        '''
        T=self.T

        a=float(self.mixingrules(comp,ll)[0])
        b=float(self.mixingrules(comp,ll)[1])
        R=self.R
        epsilon = self.objects[self.componentlist[0]].epsilon
        sigma   = self.objects[self.componentlist[0]].sigma
        pp   = P*b/(R*T)
        ppp  = (1.0 + pp)/pp
        ap   = a/(P*b**2)
        A =  pp*(epsilon + sigma - ppp)
        B =  pp**2*(ap + sigma*epsilon - (epsilon + sigma)*ppp)
        C = -pp**3*(ap + sigma*epsilon*ppp) 
        compressibilities = Utilities.cubicsolver([A, B, C])
        if state=='liquid':
            Z=min(compressibilities)
        elif state=='vapour':
            Z=max(compressibilities)
        V=Z*R*T/P
        return V
    def phibars(self,comp,state,P,ll):
        '''
        INPUT   : comp      : List of mole fractions of each species in mixture.
                  ll          : Mode, may take '0' or '1'
                  P            : the given pressure in Pa 
        CALLS   : mixingrules:gives the amix, bmix values for the compounds in the stream
                  bbars     :returns the partial parameter bbar list of compounds
                  abars      :returns the partial parameter abar list of compounds
        RETURNS : phibars      :the fugacity coefficient 
        '''
        T=self.T

        aa=float(self.mixingrules(comp,ll)[0])
        b=float(self.mixingrules(comp,ll)[1])
        R=self.R
        epsilon = self.objects[self.componentlist[0]].epsilon
        sigma   = self.objects[self.componentlist[0]].sigma
        V=self.mixturevolume(comp,state,P,ll)
        Z=V*P/(R*T)
        lnphibars={}
        phibars={}
        bbars=self.bbars(comp,ll)
        abars=self.abars(comp,ll)
        for a in self.componentlist:
            bib=bbars[a]
            aib=abars[a]
            xx=(aa/(b*R*T))/(epsilon-sigma)
            yy=1+aib/aa-bib/b
            zz=scipy.log((V+sigma*b)/(V+epsilon*b))
            lnphibars[a]=(bib/b)*(Z-1)-scipy.log((V-b)*Z/V)+xx*yy*zz
            phibars[a]=scipy.exp(lnphibars[a])
        return phibars
            
        
    def doVLEcalc(self,compositions):
        '''
        INPUT   : compositions :the liquid phase composition of compounds in the inlet
          
        CALLS   : createk     :returns the list of kp and kq value for the given compounds
                  __guesstimates: Calls the function that calculates the guess values for vapour phase mole fraction and saturation pressure and calculates self.pguess and self.yguess
                  phibars     : List of mixture fugacity co-efficients for each component in the mixture
                  createlists : Converts a list of mole fractions to dictionary
        RETURNS : endd          : List of equilibrium vapour phase mole fractions and bubble pressure in Pa.
        '''
        self.createk()
        R=self.R
        T=self.T
        x=copy.copy(compositions)
        self.__guesstimates(self.objects,compositions)
        pp=self.pguess
        yyyy=copy.copy(self.yguess)
        zzz=[]
        for i in range(len(self.componentlist)):
            zzz.append(0)
        for a in self.componentlist:
            zzz[self.componentlist.index(a)]=yyyy[a]
        zzz.append(pp)
        def eqfun(y):
            out=[]
            sum2=0.0
            aee=[]
            y=list(y)
            Pg=y[-1]
            y.pop(y.index(Pg))
            for a in self.componentlist:
                sum2=sum2+y[self.componentlist.index(a)]
            for a in self.componentlist:
                y[self.componentlist.index(a)]=y[self.componentlist.index(a)]/sum2
            gg=self.phibars(x,'liquid',Pg,1)
            hh=self.phibars(y,'vapour',Pg,1)
            for a in self.componentlist:
#                y[self.componentlist.index(a)]=y[self.componentlist.index(a)]/sum2
                aaa=y[self.componentlist.index(a)]*hh[a]-x[self.componentlist.index(a)]*gg[a]
                out.append(aaa)
            out.append(sum2-1)
            return out
        aa=scipy.optimize.fsolve(eqfun,zzz)
        aa=list(aa)
        P=aa[-1]
        aa.pop(aa.index(P))
        endd={}
        endd=self.createlists(aa,0)
        endd['Bubble Pressure in Pa']=P
        return endd
    def doflashcalc(self,feed,initcomp,finalP):
        '''
        INPUT   : feed       : Amount liquid feed initially present.
                  initcomp   : List of mole fractions of all components in input to flasher.
                  finalP     : Final Equilibrium Pressure in Pa.
                  
        CALLS   : createlists           : Converts the list of mole fractions to dictionary form.
                  CompressibilityFactors :calculates the compressibility factors at the given T and P
                  getfugacoeff           :Calculates the pure component fugacity coefficient for each species in the mixture.
                  phibars               : Returns the fugacity coefficient in mixture of each component (phibar)
        RETURNS : xx         : Liquid phase mole fractions 
                  yy         : Vapour phase mole fractions 
                  ff         : Amounts of liquid and vapour present after flashing.
        '''
        P=finalP
        T=self.T
        s=0.0
        listz=self.createlists(initcomp,0)
        guesses=[]
        o=copy.copy(self.objects)
        yyyyy={}
        n=len(self.componentlist)
        for a in range(len(self.componentlist)):
            guesses.append(1.0/n)
        for a in self.componentlist:
            z1=o[a].CompressibilityFactors(T,P)[1]
            z2=o[a].CompressibilityFactors(T,P)[0]
            z1=float(z1)
            z2=float(z2)
            zz1=o[a].getfugacoeff(z1,T,P)
            zz2=o[a].getfugacoeff(z2,T,P)
            yyyyy[a]=guesses[self.componentlist.index(a)]*(zz1/zz2)
            s+=yyyyy[a]
            
        for a in self.componentlist:
            yyyyy[a]=yyyyy[a]/s
            guesses.append(yyyyy[a])
        guesses.append(0.5)
        def flashff(x):
            n=len(x)
            x=list(x)
            alpha=x.pop(n-1)
            nn=len(x)
            summ=0
            summm=0
            c=copy.copy(self.componentlist)
            xx=[]
            yy=[]
            outt=[]
            for i in range(len(self.componentlist)):
               xx.insert(i,x[i])
               summ=summ+xx[i]
            nnn=len(self.componentlist)
            for i in range(nnn,nn):
                j=i-len(self.componentlist)
                yy.insert(j,x[i])
                summm=summm+yy[j]
            for i in range(nnn):
                xx[i]=xx[i]/summ
                yy[i]=yy[i]/summm
            phix=self.phibars(xx,'liquid',P,1)
            phiy=self.phibars(yy,'vapour',P,1)
            for a in self.componentlist:
                xyz=xx[c.index(a)]*phix[a]-yy[c.index(a)]*phiy[a]
                outt.append(xyz)
            del c[0]
            for a in c:
                xyza=xx[c.index(a)]*(1-alpha)+yy[c.index(a)]*alpha-listz[a]
                outt.append(xyza)
            outt.append(summ-1.0)
            outt.append(summm-1.0)
            return outt
        aaab=scipy.optimize.fsolve(flashff,guesses)
        aaab=list(aaab)
        alpha=aaab[-1]
        aaab.pop(aaab.index(alpha))
        xx={}
        yy={}
        for a in self.componentlist:
            xx[a]=aaab[self.componentlist.index(a)]
        n=len(aaab)
        nn=len(self.componentlist)
        for a in range(nn,n):
            j=a-nn
            yy[self.componentlist[j]]=aaab[a]
        yy['Fraction of feed vapourized']=alpha
        ff={}
        ff['Amount in liquid']=(1.0-alpha)*feed
        ff['Amount in Vapour']=alpha*feed
        return xx,yy,ff
            
        
        
#class stream:
#        def __init__(self,components,amount,comp):
#            self.amount=amount
#            self.composition={}
#            for a in components:
#                self.composition[a]=comp[components.index(a)]
#            
#class Flash(WongSandlerVLE):
#        def __init(self,components,amount,compositions,T,P,eosname):
#            WongSandlerVLE. __init__(components,eosname,T)
#            feed=stream(components,amount,compositions)
#            self.outcompliq='junk'
#            self.outcompvap='junk'
#            self.liqguess='junk'
#            self.vapguess='junk'
#            self.combinedlist='junk'
#            
#        def __setguesses(self):
#            n=len(self.componentlist)
#            nlguess=feed.amount/2.0
#            nvguess=feed.amount/2.0
#            aaa=1.0/n
#            outcompliqguess=[]
#            outcompvapguess=[]
#            for a in range(len(self.componentlist)):
#                outcompliqguess.append(aaa)
#                outcompvapguess.append(aaa)
#            self.liqguess=stream(self.componentlist,nlguess,outcompliqguess)
#            self.vapguess=stream(self.componentlist,nvguess,outcompvapguess)
#        def doflash(self):
#            self.P=P
#            def errf(x,y):
#                out=[]
#                
#                for a in self.componentlist:
#                
#            
#            
#            
#            
#            
#        
            
            
            
    
                
                
        
if __name__ == '__main__':
         def testobject(mol):
            print 'Name',mol.Name 
            print 'MolecularWeight',mol.MolecularWeight
            print 'NormalBoilingPoint',mol.NormalBoilingPoint
            print 'FreezingPoint',mol.FreezingPoint
            print 'CriticalTemperature',mol.CriticalTemperature
            print 'CriticalPressure',mol.CriticalPressure
            print 'CriticalVolume',mol.CriticalVolume
            print 'AcentricFactor',mol.AcentricFactor
            print 'ReferenceTemperature',mol.ReferenceTemperature
            print 'ReferencePressure',mol.ReferencePressure
            print 'EnthalpyFormation',mol.EnthalpyFormation
            print 'GibbsEnergyFormation',mol.GibbsEnergyFormation
            print 'EntropyReference',mol.EntropyReference
            print 'VapourPressureParameters',mol.listVapourPressureParameters
            print 'LiquidDensityParameters',mol.listLiquidDensityParameters
            print 'EnthalpyVapourizationParameters',mol.listEnthalpyVapourizationParameters
            print 'SpecificHeatCapacityParameters',mol.listSpecificHeatCapacityParameters
         mol = Molecule('n-hexane')
         testobject(mol)
