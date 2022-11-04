
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "numpy/arrayobject.h"

#include <bout/physicsmodel.hxx>
#include <smoothing.hxx>
#include <invert_laplace.hxx>
#include <derivs.hxx>



Field3D CallPythonPlugIn(Field3D n) {

  // create subarray with n and phi
  const int SIZE = 256;   // we assume square size always and take z has iit has no guard cells
  const int ND = 1;       // number of dimensions
  npy_intp dims[1];
  dims[0] = SIZE;

  double c_arr[SIZE];
  double *c_out;
  for (int i=0; i < SIZE; i++)
    c_arr[i] = 1.0;


  // Initialize Python
  Py_Initialize();
  _import_array();

  // Set path
  PySys_SetPath((wchar_t*)L"/home/jcastagna/projects/Turbulence_with_Style/PhaseII_FARSCAPE2/codes/BOUT-dev/examples/hasegawa-wakatani/");

  // Import module
  PyObject *pModule = PyImport_ImportModule("mytest");
  if(pModule == NULL){
    printf("The Module is not correctly imported \n");
  }pModule;

  // Retrive function
  PyObject *pFunc = PyObject_GetAttrString(pModule, "myabs");

  // Convert argument to Python object
  //PyObject *args = PyTuple_Pack(1,PyFloat_FromDouble(val));

  // Convert argument to Python object
  PyObject *pArray = PyArray_SimpleNewFromData(ND, dims, NPY_DOUBLE, reinterpret_cast<void*>(c_arr));
  PyArrayObject *np_arg = reinterpret_cast<PyArrayObject*>(pArray);


  // Invoke the function
  PyObject *pReturn = PyObject_CallFunctionObjArgs(pFunc, np_arg, NULL);

  // Convert it back to my type
  PyArrayObject *np_ret = reinterpret_cast<PyArrayObject*>(pReturn);

  // Convert back to C++ array and print.
  int len = PyArray_SHAPE(np_ret)[0];
  c_out = reinterpret_cast<double*>(PyArray_DATA(np_ret));


  // Free all temporary Python objects.
  Py_DECREF(pModule);
  Py_DECREF(pFunc);
  Py_DECREF(np_arg);
  Py_DECREF(np_ret);  

  // finalize
  Py_Finalize();

  Field3D n2=0.0;
  if (c_out != NULL)
  {
    n2(0,0,0)=0.0;
  }
  else
  {
    n2(0,0,0)=1.0;
  }
  return n2;
}

class HW : public PhysicsModel {
private:
  Field3D n, vort;  // Evolving density and vorticity
  Field3D phi;      // Electrostatic potential

  // Model parameters
  BoutReal alpha;      // Adiabaticity (~conductivity)
  BoutReal kappa;      // Density gradient drive
  BoutReal Dvort, Dn;  // Diffusion 
  bool modified; // Modified H-W equations?
  
  // Poisson brackets: b0 x Grad(f) dot Grad(g) / B = [f, g]
  // Method to use: BRACKET_ARAKAWA, BRACKET_STD or BRACKET_SIMPLE
  BRACKET_METHOD bm; // Bracket method for advection terms
  
  std::unique_ptr<Laplacian> phiSolver; // Laplacian solver for vort -> phi

  // Simple implementation of 4th order perpendicular Laplacian
  Field3D Delp4(const Field3D &var) {
    Field3D tmp;
    tmp = Delp2(var);
    mesh->communicate(tmp);
    tmp.applyBoundary("neumann");
    return Delp2(tmp);

    //return Delp2(var);
  }
  
protected:
  int init(bool UNUSED(restart)) {

    auto& options = Options::root()["hw"];
    alpha = options["alpha"].withDefault(1.0);
    kappa = options["kappa"].withDefault(0.1);
    Dvort = options["Dvort"].withDefault(1e-2);
    Dn = options["Dn"].withDefault(1e-2);

    modified = options["modified"].withDefault(false);

    SOLVE_FOR(n, vort);
    SAVE_REPEAT(phi);

    // Split into convective and diffusive parts
    setSplitOperator();
    
    phiSolver = Laplacian::create();
    phi = 0.; // Starting phi
    
    // Use default flags 
    
    // Choose method to use for Poisson bracket advection terms
    switch(options["bracket"].withDefault(0)) {
    case 0: {
      bm = BRACKET_STD; 
      output << "\tBrackets: default differencing\n";
      break;
    }
    case 1: {
      bm = BRACKET_SIMPLE; 
      output << "\tBrackets: simplified operator\n";
      break;
    }
    case 2: {
      bm = BRACKET_ARAKAWA; 
      output << "\tBrackets: Arakawa scheme\n";
      break;
    }
    case 3: {
      bm = BRACKET_CTU; 
      output << "\tBrackets: Corner Transport Upwind method\n";
      break;
    }
    default:
      output << "ERROR: Invalid choice of bracket method. Must be 0 - 3\n";
      return 1;
    }
    
    return 0;
  }

  int convective(BoutReal UNUSED(time)) {
    // Non-stiff, convective part of the problem
    
    // Solve for potential
    phi = phiSolver->solve(vort, phi);
    
    // Communicate variables
    mesh->communicate(n, vort, phi);
    
    // Modified H-W equations, with zonal component subtracted from resistive coupling term
    Field3D nonzonal_n = n;
    Field3D nonzonal_phi = phi;
    if(modified) {
      // Subtract average in Y and Z
      nonzonal_n -= averageY(DC(n));
      nonzonal_phi -= averageY(DC(phi));
    }
    
    ddt(n) = -bracket(phi, n, bm) + alpha*(nonzonal_phi - nonzonal_n) - kappa*DDZ(phi);
    
    ddt(vort) = -bracket(phi, vort, bm) + alpha*(nonzonal_phi - nonzonal_n);
  
    return 0;
  }
  
  int diffusive(BoutReal UNUSED(time)) {
    // Diffusive terms
    mesh->communicate(n, vort);

    Field3D f=0.0;

    Field3D nLES = CallPythonPlugIn(f);

    ddt(n) = -Dn*Delp4(n) + nLES;
    ddt(vort) = -Dvort*Delp4(vort) + nLES;
    return 0;
  }
};

// Define a main() function
BOUTMAIN(HW);
