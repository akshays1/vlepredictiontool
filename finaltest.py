import mastersource
import Utilities
#import scipy
equation=mastersource.WongSandlerVLE(['n-pentane','n-hexane','n-heptane'],'PR',346.16)
#yequation=mastersource.Molecule_UNIQUAC(['aceticacid','water'])
#print equation.VapourPressureEOS(300)
#equation.adjustparams(10)
#zg,zl= equation.CompressibilityFactors(200,1000)
#print zl-1000*equation.b/(225.3*equation.R)
#print 1/(equation.VlsatEOS(420))
#print equation.VapourPressureEOS(266)
#print equation.LiquidDensity(420,0)
#
#print equation.VapourPressureEOS(300)
#print equation.VapourPressure(300,0)

#print equation.EnthalpyVapourization(350,0)
#print equation.HvapEOS(350)
#print equation.b*1000/(8314*600)5 
#print equation.yguess
#print equation.listx
#print equation.pguess
#print equation.getgexcomb(300,[0.5,0.5])
#print equation.getD([0.6,0.4])
#print equation.mixingrules([0.5,0.5])
#print equation.phibars([0.9,0.1],'vapour',100000)
#print equation.doVLEcalc([0.5,0.5])
#print equation.objects['ethanol'].NormalBoilingPoint
#print equation.doflashcalc(1,[0.25,0.45,0.30],101300)
print equation.doVLEcalc([0.072,0.342,0.586])
#print equation.getGammaRes(342.8,[0.564,1-0.564],0)
