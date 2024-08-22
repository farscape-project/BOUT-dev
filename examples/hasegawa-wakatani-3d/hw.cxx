/// 3D simulations of HW
///
/// This version uses indexed operators
/// which reduce the number of loops over the domain
///
/// GPU processing is enabled if BOUT_HAS_CUDA is defined
/// Profiling markers and ranges are set if USE_NVTX is defined
/// Based on Ben Dudson, Steven Glenn code, Yining Qin update 0521-2020
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <tuple>

#include <iostream>

#include <bout/invert_laplace.hxx>
#include <bout/physicsmodel.hxx>
#include <bout/single_index_ops.hxx>

#define DISABLE_RAJA 0
#include <bout/rajalib.hxx>


#include <chrono>
using namespace std::chrono;




//----------------------------------------------------Start StylES functions------------------------------------

void initPythonModule(PyObject **pModule, PyObject **pInitFlow, PyObject **pFindLESTerms, PyObject **pWritePoissonDNS) {


  // Initialize Python interpreter
  Py_Initialize();
  _import_array();


  // set Python system path
  PyObject *sys_path = PySys_GetObject("path");
  PyList_Append(sys_path, PyUnicode_FromString("../../../../StylES/bout_interfaces/"));

  // Import Python module
  *pModule = PyImport_ImportModule("pBOUT");
  if(*pModule == NULL) {
    PyErr_Print();
    fprintf(stderr, "Import Python module failed!\n");
  }


  // Load Python functions and execute it
  *pInitFlow = PyObject_GetAttrString(*pModule, "initFlow");

  if (!(*pInitFlow) || !PyCallable_Check(*pInitFlow)) {
    Py_DECREF(pModule);
    PyErr_Print();
    fprintf(stderr, "Python init function not found!\n");
  }


  // Load Python functions and execute it
  *pFindLESTerms = PyObject_GetAttrString(*pModule, "findLESTerms");

  if (!(*pFindLESTerms) || !PyCallable_Check(*pFindLESTerms)) {
    Py_DECREF(pModule);
    PyErr_Print();
    fprintf(stderr, "Python function not found!\n");
  }


  // Load Python functions and execute it
  *pWritePoissonDNS = PyObject_GetAttrString(*pModule, "writePoissonDNS");

  if (!(*pWritePoissonDNS) || !PyCallable_Check(*pWritePoissonDNS)) {
    Py_DECREF(pModule);
    PyErr_Print();
    fprintf(stderr, "Python function not found!\n");
  }


  return;

}




void closePythonModule(PyObject **pModule, PyObject **pInitFlow, PyObject **pFindLESTerms, PyObject **pWritePoissonDNS) {

  // shutdown Python interpreter
  Py_DECREF(pInitFlow);
  Py_DECREF(pFindLESTerms);
  Py_DECREF(pWritePoissonDNS);
  Py_DECREF(pModule);

  if (Py_FinalizeEx() < 0) {
    PyErr_Print();
    fprintf(stderr, "Failed to shutdown Python!");
  }

  return;

}




double* initFlow(double dx, double dy, Field3D n, Field3D phi, Field3D vort, PyObject *pModule, PyObject *pInitFlow) {

  // local variables
  PyObject *pValue = NULL;
  PyObject *pArray = NULL;  
  PyArrayObject *pArgs = NULL;
  PyArrayObject *pReturn = NULL;  

  const int SIZEX = n.getNx()-4;
  const int SIZEY = n.getNy()-4;
  const int SIZEZ = n.getNz();
  const int SIZET = 3*SIZEX*SIZEY*SIZEZ;
  const int ND    = 1;

  int i;
  int j;
  int k;
  int cont;

  double* fLES;
  double* pLES = new double[2+SIZET];

  if (!pLES) {
      fprintf(stderr, "Out of memory when allocating array pLES!\n");
  }

  npy_intp dims[1]{SIZET};




  // pass n and vort arrays to pLES
  pLES[0] = dx;
  pLES[1] = dy;

  cont=2;
  for(i=2; i<SIZEX+2; i++)   // we assume 2 guards cells in x-direction
    for(j=2; j<SIZEY+2; j++)
      for(k=0; k<SIZEZ; k++){
        pLES[cont++] = n(i,j,k);
        pLES[cont++] = phi(i,j,k);
        pLES[cont++] = vort(i,j,k);
      }



  // convert to numpy array   
  pArray = PyArray_SimpleNewFromData(ND, dims, NPY_DOUBLE, reinterpret_cast<void*>(pLES));
  if (pArray)
  {

    // create arguments
    pArgs = reinterpret_cast<PyArrayObject*>(pArray);
    if (pArgs!=NULL) {

      // call function
      pValue = PyObject_CallFunctionObjArgs(pInitFlow, pArray, NULL);

      if (pValue!=NULL) {

        pReturn = reinterpret_cast<PyArrayObject*>(pValue);
        printf("Dimensions of returned n array are: %d\n", PyArray_NDIM(pReturn));

        // convert result back to C++
        fLES = reinterpret_cast<double*>(PyArray_DATA(pReturn));

        // decrement Python object counter
        // Py_DECREF(pReturn);
        // Py_DECREF(pValue);
        // Py_DECREF(pArgs);
        // Py_DECREF(pArray);

      } else {
        Py_DECREF(pValue);
        Py_DECREF(pArgs);
        Py_DECREF(pInitFlow);
        Py_DECREF(pModule);
        PyErr_Print();
        fprintf(stderr, "Call to Python function failed!\n");
      }
    
    } else {
      PyErr_Print();
      fprintf(stderr, "Arguments not created!\n");
    }

  } else {
    PyErr_Print();
    fprintf(stderr, "Array not created!\n");
  }


  delete [] pLES;

  return fLES;

}




double* findLESTerms(const int pStep, const int pStepStart, const double dx, const double simtime, const Field3D& n, Field3D& phi, Field3D& vort, Field3D& pPhiVort, Field3D& pPhiN,
  const bool implicitStylES, PyObject *pModule, PyObject *pFindLESTerms) {

  // local variables
  PyObject *pValue = NULL;
  PyObject *pArray = NULL;  
  PyArrayObject *pArgs = NULL;
  PyArrayObject *pReturn = NULL;  

  const int SIZEX = n.getNx()-4;
  const int SIZEY = n.getNy()-4;
  const int SIZEZ = n.getNz();
  const int SIZE  = SIZEX*SIZEY*SIZEZ;
  const int SIZET = 4+3*SIZE;
  const int ND    = 1;

  int i;
  int j;
  int k;
  int cont;

  double* fLES;
  double* pLES = new double[SIZET];

  if (!pLES) {
      fprintf(stderr, "Out of memory when allocating array pLES!\n");
  }

  npy_intp dims[1]{SIZET};



  // pass n and vort arrays to pLES
  pLES[0] = double(pStep);
  pLES[1] = double(pStepStart);
  pLES[2] = dx;
  pLES[3] = simtime;


  cont=4;
  for(int i=2; i<n.getNx()-2; i++)   // we assume 2 guards cells in x-direction
    for(int j=2; j<n.getNy()-2; j++)
      for(int k=0; k<n.getNz(); k++){
        pLES[cont + 0*SIZE] = n(i,j,k);
        pLES[cont + 1*SIZE] = phi(i,j,k);
        pLES[cont + 2*SIZE] = vort(i,j,k);
        cont = cont+1;
      }


  // convert to numpy array   
  pArray = PyArray_SimpleNewFromData(ND, dims, NPY_DOUBLE, reinterpret_cast<void*>(pLES));
  if (pArray)
  {

    // create arguments
    pArgs = reinterpret_cast<PyArrayObject*>(pArray);
    if (pArgs!=NULL) {

      // call function
      pValue = PyObject_CallFunctionObjArgs(pFindLESTerms, pArray, NULL);

      if (pValue!=NULL) {

        pReturn = reinterpret_cast<PyArrayObject*>(pValue);
        //printf("Dimensions of returned n array are: %d\n", PyArray_NDIM(pReturn));

        // convert result back to C++
        fLES = reinterpret_cast<double*>(PyArray_DATA(pReturn));

        // decrement Python object counter
        // Py_DECREF(pReturn);
        // Py_DECREF(pValue);
        // Py_DECREF(pArgs);
        // Py_DECREF(pArray);

      } else {
        Py_DECREF(pValue);
        Py_DECREF(pArgs);
        Py_DECREF(pFindLESTerms);
        Py_DECREF(pModule);
        PyErr_Print();
        fprintf(stderr, "Call to Python function failed!\n");
      }
    
    } else {
      PyErr_Print();
      fprintf(stderr, "Arguments not created!\n");
    }

  } else {
    PyErr_Print();
    fprintf(stderr, "Array not created!\n");
  }


  delete [] pLES;

  return fLES;

}




double* writePoissonDNS(int pStep, int pStepStart, double dx, double simtime, Field3D n, Field3D phi, Field3D vort, Field3D pPhiVort, Field3D pPhiN,
  PyObject *pModule, PyObject *pWritePoissonDNS) {

  // local variables
  PyObject *pValue = NULL;
  PyObject *pArray = NULL;  
  PyArrayObject *pArgs = NULL;
  PyArrayObject *pReturn = NULL;  

  const int SIZEX  = n.getNx()-4;
  const int SIZEY  = n.getNy()-4;
  const int SIZEZ  = n.getNz();
  const int SIZE   = SIZEX*SIZEY*SIZEZ;  
  const int SIZET = 4+5*SIZE;
  const int ND    = 1;

  int i;
  int j;
  int k;
  int cont;

  double* fLES;
  double* pLES = new double[SIZET];

  if (!pLES) {
      fprintf(stderr, "Out of memory when allocating array pLES!\n");
  }

  npy_intp dims[1]{SIZET};



  // pass n and vort arrays to pLES
  pLES[0] = double(pStep);
  pLES[1] = double(pStepStart);
  pLES[2] = dx;
  pLES[3] = simtime;

  cont=4;
  for(int i=2; i<n.getNx()-2; i++)   // we assume 2 guards cells in x-direction
    for(int j=2; j<n.getNy()-2; j++)
      for(int k=0; k<n.getNz(); k++){
        pLES[cont + 0*SIZE] = n(i,j,k);
        pLES[cont + 1*SIZE] = phi(i,j,k);
        pLES[cont + 2*SIZE] = vort(i,j,k);
        pLES[cont + 3*SIZE] = pPhiVort(i,j,k);
        pLES[cont + 4*SIZE] = pPhiN(i,j,k);                
        cont = cont+1;
      }


  // convert to numpy array   
  pArray = PyArray_SimpleNewFromData(ND, dims, NPY_DOUBLE, reinterpret_cast<void*>(pLES));
  if (pArray)
  {

    // create arguments
    pArgs = reinterpret_cast<PyArrayObject*>(pArray);
    if (pArgs!=NULL) {

      // call function
      pValue = PyObject_CallFunctionObjArgs(pWritePoissonDNS, pArray, NULL);

      if (pValue!=NULL) {

        pReturn = reinterpret_cast<PyArrayObject*>(pValue);
        //printf("Dimensions of returned n array are: %d\n", PyArray_NDIM(pReturn));

        // convert result back to C++
        fLES = reinterpret_cast<double*>(PyArray_DATA(pReturn));

        // decrement Python object counter
        // Py_DECREF(pReturn);
        // Py_DECREF(pValue);
        // Py_DECREF(pArgs);
        // Py_DECREF(pArray);

      } else {
        Py_DECREF(pValue);
        Py_DECREF(pArgs);
        Py_DECREF(pWritePoissonDNS);
        Py_DECREF(pModule);
        PyErr_Print();
        fprintf(stderr, "Call to Python function failed!\n");
      }
    
    } else {
      PyErr_Print();
      fprintf(stderr, "Arguments not created!\n");
    }

  } else {
    PyErr_Print();
    fprintf(stderr, "Array not created!\n");
  }


  delete [] pLES;

  return fLES;

}




void initPythonModule(PyObject **pModule, PyObject **pInitFlow, PyObject **pFindLESTerms) {


  // Initialize Python interpreter
  Py_Initialize();
  _import_array();


  // set Python system path
  PyObject *sys_path = PySys_GetObject("path");
  PyList_Append(sys_path, PyUnicode_FromString("../../../../StylES/bout_interfaces/"));

  // Import Python module
  *pModule = PyImport_ImportModule("pBOUT");
  if(*pModule == NULL) {
    PyErr_Print();
    fprintf(stderr, "Import Python module failed!\n");
  }


  // Load Python functions and execute it
  *pInitFlow = PyObject_GetAttrString(*pModule, "initFlow");

  if (!(*pInitFlow) || !PyCallable_Check(*pInitFlow)) {
    Py_DECREF(pModule);
    PyErr_Print();
    fprintf(stderr, "Python init function not found!\n");
  }


  // Load Python functions and execute it
  *pFindLESTerms = PyObject_GetAttrString(*pModule, "findLESTerms");

  if (!(*pFindLESTerms) || !PyCallable_Check(*pFindLESTerms)) {
    Py_DECREF(pModule);
    PyErr_Print();
    fprintf(stderr, "Python function not found!\n");
  }

  return;

}




void closePythonModule(PyObject **pModule, PyObject **pInitFlow, PyObject **pFindLESTerms) {

  // shutdown Python interpreter
  Py_DECREF(pInitFlow);
  Py_DECREF(pFindLESTerms);
  Py_DECREF(pModule);

  if (Py_FinalizeEx() < 0) {
    PyErr_Print();
    fprintf(stderr, "Failed to shutdown Python!");
  }

  return;

}

//----------------------------------------------------End StylES functions------------------------------------




class HW3D : public PhysicsModel {
private:
  Field3D n, vort; // Evolving density and vorticity
  Field3D phi;     // Electrostatic potential

  // Model parameters
  BoutReal alpha;                       // Adiabaticity (~conductivity)
  BoutReal kappa;                       // Density gradient drive
  BoutReal Dvort, Dn;                   // Diffusion
  std::unique_ptr<Laplacian> phiSolver; // Laplacian solver for vort -> phi


  // StylES variables
  PyObject *pModule;
  PyObject *pInitFlow;
  PyObject *pFindLESTerms;
  PyObject *pWritePoissonDNS;

  int pStep      = 0;
  int pStepStart = 0;  // for pStep<pStepStart you have a DNS.

  double deltax;
  double deltaz;
  double psimtime = 0.0;

  bool profile_StylES = false;
  bool implicitStylES = false;

  Field3D pPhiVort;
  Field3D pPhiN;

  Field3D tPhiVort;
  Field3D tPhiN;
  

  int totCount = 0;
  std::chrono::time_point<std::chrono::high_resolution_clock> timeStart;
  std::chrono::time_point<std::chrono::high_resolution_clock> timeStop;
  std::chrono::microseconds totDuration1{0};
  std::chrono::microseconds totDuration2{0};


public:
  // Note: rhs() function must be public, so that RAJA can use CUDA

  int init(bool UNUSED(restart)) override {

    auto& options = Options::root()["hw"];
    alpha = options["alpha"].withDefault(1.0);
    kappa = options["kappa"].withDefault(0.1);
    Dvort = options["Dvort"].doc("Vorticity diffusion (normalised)").withDefault(1e-2);
    Dn = options["Dn"].doc("Density diffusion (normalised)").withDefault(1e-2);

    SOLVE_FOR(n, vort);
    SAVE_REPEAT(phi);
    phiSolver = Laplacian::create(nullptr, CELL_CENTRE, mesh, solver);

    phi = 0.; // Starting phi


    // StylES initializations
    const int SIZEX = n.getNx()-4;
    const int SIZEY = n.getNy()-4;
    const int SIZEZ = n.getNz();
    const int SIZE  = SIZEX*SIZEY*SIZEZ;  

    pPhiVort = 0.;
    pPhiN = 0.;

    tPhiVort = 0.;
    tPhiN = 0.;

    if (pStep==0) {
      initPythonModule(&pModule, &pInitFlow, &pFindLESTerms, &pWritePoissonDNS);
    }


    CELL_LOC outloc = n.getLocation();
    Coordinates *metric = phi.getCoordinates(outloc);

    deltax = metric->dx(0,0,0);
    deltaz = metric->dz(0,0,0);

    double *npv;

    npv = initFlow(deltax, deltaz, n, phi, vort, pModule, pInitFlow);

    // return npv
    int cont=0;
    for(int i=2; i<n.getNx()-2; i++)   // we assume 2 guards cells in x-direction
      for(int j=2; j<n.getNy()-2; j++)
        for(int k=0; k<n.getNz(); k++){
          n(i,j,k)    = npv[cont + 0*SIZE];
          phi(i,j,k)  = npv[cont + 1*SIZE];
          vort(i,j,k) = npv[cont + 2*SIZE];
          cont = cont+1;
        }

    // // close Python console
    // closePythonModule(&pModule, &pInitFlow, &pFindLESTerms, &pWritePoissonDNS);


    // Communicate variables
    mesh->communicate(n, phi, vort);

    return 0;
  }




  int rhs(BoutReal time) override {

    double *rLES;

    // Solve for potential
    if (pStep>0){
      phi = phiSolver->solve(vort, phi);
    }

    Field3D phi_minus_n = phi - n;
    // Communicate variables
    mesh->communicate(n, vort, phi, phi_minus_n);

    // Create accessors which enable fast access
    auto n_acc = FieldAccessor<>(n);
    auto vort_acc = FieldAccessor<>(vort);
    auto phi_acc = FieldAccessor<>(phi);
    auto phi_minus_n_acc = FieldAccessor<>(phi_minus_n);

    const int SIZEX  = n.getNx()-4;
    const int SIZEY  = n.getNy()-4;
    const int SIZEZ  = n.getNz();
    const int SIZE   = SIZEX*SIZEY*SIZEZ;  

    // find brackets term (for explicit) or subgrid scale terms (for implicit) via StylES
    if (pStep>=pStepStart)
    {
      double simtime = time;
      output_progress.print("\r");
      rLES = findLESTerms(pStep, pStepStart, deltax, simtime, n, phi, vort, pPhiVort, pPhiN, implicitStylES, pModule, pFindLESTerms);
      int LES_it = int(rLES[0]);
      int cont=1;

      for(int i=2; i<n.getNx()-2; i++)   // we assume 2 guards cells in x-direction
        for(int j=2; j<n.getNy()-2; j++)
          for(int k=0; k<n.getNz(); k++){
            pPhiVort(i,j,k) = rLES[cont + 0*SIZE];
            pPhiN(i,j,k)    = rLES[cont + 1*SIZE];
            cont = cont+1;
          }
    }
    else{
      BOUT_FOR_RAJA(i, n.getRegion("RGN_NOBNDRY"), CAPTURE(alpha, kappa, Dn, Dvort)) {
        pPhiVort[i] = bracket(phi_acc, vort_acc, i);
        pPhiN[i]    = bracket(phi_acc, n_acc, i);
      }
    }


    // integrate
    BOUT_FOR_RAJA(i, n.getRegion("RGN_NOBNDRY"), CAPTURE(alpha, kappa, Dn, Dvort)) {
      BoutReal div_current = alpha * Div_par_Grad_par(phi_minus_n_acc, i);

      ddt(n_acc)[i] = -pPhiN[i] - div_current - kappa * DDZ(phi_acc, i) + Dn * Delp2(n_acc, i);

      ddt(vort_acc)[i] =  -pPhiVort[i] - div_current + Dvort * Delp2(vort_acc, i);
    }

    pStep++;
    return 0;
  }
};

// Define a main() function
BOUTMAIN(HW3D);
