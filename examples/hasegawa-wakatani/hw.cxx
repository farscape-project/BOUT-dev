#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include<tuple>

#include <bout/physicsmodel.hxx>
#include <smoothing.hxx>
#include <invert_laplace.hxx>
#include <derivs.hxx>



void initPythonModule(PyObject **pModule, PyObject **pInitFlow, PyObject **pFindLESTerms) {


  // Initialize Python interpreter
  Py_Initialize();
  _import_array();


  // set Python system path
  PyObject *sys_path = PySys_GetObject("path");
  PyList_Append(sys_path, PyUnicode_FromString("/lustre/scafellpike/local/HT04543/jxc06/jxc74-jxc06/projects/Turbulence_with_Style/PhaseII_FARSCAPE2/codes/StylES/bout_interfaces/"));

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



double* initFlow(Field3D n, Field3D phi, Field3D vort, PyObject *pModule, PyObject *pInitFlow) {

  // local variables
  PyObject *pValue = NULL;
  PyObject *pArray = NULL;  
  PyArrayObject *pArgs = NULL;
  PyArrayObject *pReturn = NULL;  

  const int SIZE  = n.getNz();
  const int SIZE2 = 3*SIZE*SIZE;
  const int ND    = 1;

  int i;
  int j;
  int k;
  int cont;

  double* fLES;
  double* pLES = new double[SIZE2];

  if (!pLES) {
      fprintf(stderr, "Out of memory when allocating array pLES!\n");
  }

  npy_intp dims[1]{SIZE2};




  // pass n and vort arrays to pLES
  cont=0;
  for(i=2; i<SIZE+2; i++)   // we assume 2 guards cells in x-direction
    for(j=0; j<1; j++)
      for(k=0; k<SIZE; k++){
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


double* findLESTerms(Field3D n, Field3D phi, Field3D vort, PyObject *pModule, PyObject *pFindLESTerms) {

  // local variables
  PyObject *pValue = NULL;
  PyObject *pArray = NULL;  
  PyArrayObject *pArgs = NULL;
  PyArrayObject *pReturn = NULL;  

  const int SIZE  = n.getNz();
  const int SIZE2 = 3*SIZE*SIZE;
  const int ND    = 1;

  int i;
  int j;
  int k;
  int cont;

  double* fLES;
  double* pLES = new double[SIZE2];

  if (!pLES) {
      fprintf(stderr, "Out of memory when allocating array pLES!\n");
  }

  npy_intp dims[1]{SIZE2};




  // pass n and vort arrays to pLES
  cont=0;
  for(i=2; i<SIZE+2; i++)   // we assume 2 guards cells in x-direction
    for(j=0; j<1; j++)
      for(k=0; k<SIZE; k++){
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



class HW : public PhysicsModel {
private:
  Field3D n, vort;  // Evolving density and vorticity
  Field3D phi;      // Electrostatic potential

  // Python variables
  PyObject *pModule;
  PyObject *pInitFlow;
  PyObject *pFindLESTerms;
  int pStep=0;

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
    




    if (pStep==0) {
      initPythonModule(&pModule, &pInitFlow, &pFindLESTerms);
    }


    // double *npv;

    // npv = initFlow(n, phi, vort, pModule, pInitFlow);

    // // return npv
    // int cont=0;
    // double minN= 10000.0;
    // double maxN=-10000.0;
    // double minP= 10000.0;
    // double maxP=-10000.0;
    // double minV= 10000.0;
    // double maxV=-10000.0;

    // for(int i=2; i<n.getNz()+2; i++)   // we assume 2 guards cells in x-direction
    //   for(int j=0; j<1; j++)
    //     for(int k=0; k<n.getNz(); k++){
    //       n(i,j,k)    = npv[cont++];
    //       phi(i,j,k)  = npv[cont++];
    //       vort(i,j,k) = npv[cont++];
    //       minN = std::min(minN,n(i,j,k));
    //       maxN = std::max(maxN,n(i,j,k));
    //       minP = std::min(minP,phi(i,j,k));
    //       maxP = std::max(maxP,phi(i,j,k));
    //       minV = std::min(minV,vort(i,j,k));
    //       maxV = std::max(maxV,vort(i,j,k));
    //     }

    // printf("%f %f \n", minN, maxN);
    // printf("%f %f \n", minP, maxP);
    // printf("%f %f \n", minV, maxV);


    // // close Python console
    // closePythonModule(&pModule, &pInitFlow, &pFindLESTerms);


    // Communicate variables
    mesh->communicate(n, phi, vort);


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
    pStep++;
    
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

    double *rLES;
    Field3D Dpyvx=0.0;
    Field3D Dpxvy=0.0;
    Field3D Dpynx=0.0;
    Field3D Dpxny=0.0;

    double minN= 10000.0;
    double maxN=-10000.0;
    double minP= 10000.0;
    double maxP=-10000.0;
    double minV= 10000.0;
    double maxV=-10000.0;
    double minX= 10000.0;
    double maxX=-10000.0;

    if (pStep>=0){
      rLES = findLESTerms(n, phi, vort, pModule, pFindLESTerms);
      int N_LES = n.getNz();
      int cont=0;
      for(int i=2; i<n.getNz()+2; i++)   // we assume 2 guards cells in x-direction
        for(int j=0; j<1; j++)
          for(int k=0; k<n.getNz(); k++){
            Dpyvx(i,j,k) = rLES[cont + 0*N_LES*N_LES];
            Dpxvy(i,j,k) = rLES[cont + 1*N_LES*N_LES];
            Dpynx(i,j,k) = rLES[cont + 2*N_LES*N_LES];
            Dpxny(i,j,k) = rLES[cont + 3*N_LES*N_LES];
            cont = cont+1;
            minN = std::min(minN,Dpyvx(i,j,k));
            maxN = std::max(maxN,Dpyvx(i,j,k));
            minP = std::min(minP,Dpxvy(i,j,k));
            maxP = std::max(maxP,Dpxvy(i,j,k));
            minV = std::min(minV,Dpynx(i,j,k));
            maxV = std::max(maxV,Dpynx(i,j,k));
            minX = std::min(minX,Dpxny(i,j,k));
            maxX = std::max(maxX,Dpxny(i,j,k));
          }
    }

    // printf("%f %f \n", minN, maxN);
    // printf("%f %f \n", minP, maxP);
    // printf("%f %f \n", minV, maxV);
    // printf("%f %f \n", minX, maxX);    

    ddt(n) = -Dn*Delp4(n) + Dpyvx + Dpxvy;
    ddt(vort) = -Dvort*Delp4(vort) + Dpynx + Dpxny;

    return 0;
  }
};



// Define a main() function
BOUTMAIN(HW);
