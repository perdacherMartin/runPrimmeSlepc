//
//  Created by Martin Perdacher on 2013-08-27.
//  Copyright (c) 2013 . All rights reserved.
//

static char help[] = "Calculating eigenvalues and vectors with Lapack, SLEPc. Storing in PetscBinary format\n";

#include <stdlib.h>
#include "slepceps.h"
#include "mkl.h"
#include "mkl_cblas.h"
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <complex>
#include "petsctime.h"

#define REPETITIONS 5

typedef MKL_Complex16 T_complex;

T_complex* ReduceGeneralizedToStandardform(T_complex *Hmat,T_complex *Smat, PetscInt const n);
PetscErrorCode SolveH(T_complex *H, PetscInt const n, PetscInt const nev, PetscReal *lambda, T_complex *y, char* const relErrorLapackFilename, char* const timeFilename);
T_complex* ReadSequentiallyFromFile(char const *filein, PetscInt const n);
PetscErrorCode TransposeHermitian(T_complex *matrix, PetscInt const n);
PetscErrorCode WriteToPetscBinary(PetscBool const doTranspose, T_complex *matrix, char* const filename, PetscInt const n);
PetscErrorCode SolveWithSlepc(T_complex *Hmat, T_complex *Smat, char* const relErrorSlepcFilename, 
		PetscInt const n, char* const evecSlepcFilename, char* const timeFilename);
PetscErrorCode SetUpPetscMatrix(T_complex *cmatrix, Mat matrix, PetscInt const n);

PetscErrorCode LapackRobust(T_complex *cmatrix, PetscInt const n, char* const timeFilename, PetscInt const nev, PetscReal *lambda, T_complex *y);
PetscErrorCode LapackAll(T_complex *cmatrix, PetscInt const n, char* const timeFilename, PetscReal *lambda, T_complex *y);

