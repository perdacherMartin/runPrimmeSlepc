//
//  driver
//
//  Created by Martin Perdacher on 2013-09-21.
//  Copyright (c) 2013 . All rights reserved.
//
// example call  ./driver -stdFile std/CaFe2As2.small.standard.001002.dat -evecGuess lapack/CaFe2As2.small.lapack.001001.dat -random

static char help[] = "Slepc driver for testing the primme package\n";

#include "slepceps.h"
#include "petsctime.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
	PetscErrorCode ierr;
	FILE *fp;
	PetscInt nconv,n,m,*idxm,*idxn,i,j,blockSize=1, nev,ncv,mpd;
	PetscReal error;
	PetscScalar *scalars,zero=0.0, kr, ki;
	Mat matrix, evecGuess;
	EPS eps;
	Vec xi, xr;
	PetscLogDouble t1,t2;
	PetscViewer viewer;
	PetscBool isRandom=PETSC_FALSE;
	char stProFilename[PETSC_MAX_PATH_LEN]="", evecGuessFilename[PETSC_MAX_PATH_LEN]="", timingFile[PETSC_MAX_PATH_LEN]="";
	SlepcInitialize(&argc,&argv,(char*)0,help);
	EPSPRIMMEMethod method;
	
	ierr = PetscOptionsGetString(PETSC_NULL,"-stdFile",stProFilename,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr); 
	
// loading reduced standard problem from file	
	ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,stProFilename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
	ierr = MatCreate(PETSC_COMM_WORLD, &matrix);CHKERRQ(ierr);
	ierr = MatSetFromOptions(matrix);CHKERRQ(ierr);
	ierr = MatLoad(matrix,viewer);CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
	
	ierr = PetscOptionsGetString(PETSC_NULL,"-evecGuess",evecGuessFilename,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);	
	ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,evecGuessFilename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
	ierr = MatCreate(PETSC_COMM_WORLD, &evecGuess);CHKERRQ(ierr);
	ierr = MatSetFromOptions(evecGuess);CHKERRQ(ierr);
	ierr = MatLoad(evecGuess,viewer);CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
	
// set up vectors
	ierr = MatGetSize(matrix,&m,&n);
	ierr = PetscMalloc(n*sizeof(PetscScalar), &scalars);
	ierr = PetscMalloc(n*sizeof(PetscInt),&idxm);
	ierr = PetscMalloc(n*sizeof(PetscInt),&idxn);
	for ( i = 0 ; i < n ; ++i ){ idxn[i] = i; }
	
	ierr = MatGetVecs(matrix,NULL,&xr);CHKERRQ(ierr);
	ierr = MatGetVecs(matrix,NULL,&xi);CHKERRQ(ierr);
	
// set up matrix for storing result
//	ierr = MatCreateDense(PETSC_COMM_WORLD,m,n, m, n, NULL, &primmeY);CHKERRQ(ierr);
//	ierr = MatGetOwnershipRange(*mat,&loc_start,&loc_end);
	
//    for (i=loc_start; i<loc_end; i++) {
//      MatSetValues(*mat,1,&i,1,&j,&zero,INSERT_VALUES); 
//    }
//    MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);
//    MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY);	
	
// setup eps
	ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);
	ierr = EPSSetType(eps, EPSPRIMME);
	ierr = EPSSetOperators(eps,matrix,NULL);CHKERRQ(ierr);

// standard value 136, will be overridden with -eps_nev	
	nev = 14;
	blockSize = 1;
	ierr = EPSPRIMMESetBlockSize(eps,blockSize);
	ierr = EPSSetDimensions(eps, nev, PETSC_DECIDE, PETSC_DECIDE ); 
	
	ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr);
	ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
	
// -eps_primme_method 
//        gd_plusk,  	
	ierr = EPSPRIMMESetMethod(eps, EPS_PRIMME_DEFAULT_MIN_TIME);
	// ierr = EPSPRIMMESetMethod(eps, EPS_PRIMME_JDQMR);
	// ierr = EPSPRIMMESetMethod(eps, EPS_PRIMME_DEFAULT_MIN_TIME);
	
//	-eps_conv_abs 	- Sets the absolute convergence test
//	-eps_conv_eig 	- Sets the convergence test relative to the eigenvalue
//	-eps_conv_norm 	- Sets the convergence test relative to the matrix norms
	ierr = EPSSetConvergenceTest(eps,EPS_CONV_ABS);

//  override eps with command line arguments
	ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);
	ierr = EPSSetUp(eps);
	
	ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);	
	ierr = PetscOptionsGetBool(NULL,"-random",&isRandom,NULL);
	if ( isRandom ){
		ierr = EPSPRIMMESetRandomGuesses_PRIMME(eps);		
	}else{
		ierr = EPSPRIMMESetInitialGuesses_PRIMME(eps, evecGuess);		
	}
	
	
	// check parameters before solve call
	ierr = EPSPRIMMEGetBlockSize(eps, &blockSize);
	ierr = EPSGetDimensions(eps,&nev,&ncv,&mpd);
	//ierr = PetscPrintf(PETSC_COMM_WORLD,"nev: %d, ncv: %d, mpd: %d \n", nev, ncv, mpd);	
	ierr = EPSPRIMMEGetMethod(eps,&method);
	
	ierr = PetscTime(&t1);CHKERRQ(ierr);
	ierr = EPSSolve(eps);CHKERRQ(ierr);
	ierr = PetscTime(&t2);CHKERRQ(ierr);
	
	
// get eigenvectors and write them to file 	
	ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
	
//    for (i=0;i<nconv;i++) {
//		printf("%d\n",i);
//		ierr = EPSGetEigenpair(eps,i,&kr,&ki,xr,xi);CHKERRQ(ierr);
//    	ierr = VecView(xr,viewer);CHKERRQ(ierr); // real
//    	ierr = VecView(xi,viewer);CHKERRQ(ierr); // imaginary
//
//		ierr = EPSComputeRelativeError(eps,i,&error);CHKERRQ(ierr);
//    }	
	
	ierr = PetscOptionsGetString(PETSC_NULL,"-timingFile",timingFile,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr); 
	
	if ( strcmp(timingFile,"") == 0  ){
		if ( method == EPS_PRIMME_GD_PLUSK ){
			ierr = PetscPrintf(PETSC_COMM_WORLD,"method used: GD+K\n");		
		}else if ( method == EPS_PRIMME_JDQMR )	{
			ierr = PetscPrintf(PETSC_COMM_WORLD,"method used: JDQMR\n");
		}else {
			ierr = PetscPrintf(PETSC_COMM_WORLD,"other method used then GD+K or JDQMR.");
		}

		printf("time spend: %f\n", t2-t1);
		ierr = PetscPrintf(PETSC_COMM_WORLD , "primme blocksize: %d\n", blockSize);
		if ( isRandom ){
			ierr = PetscPrintf(PETSC_COMM_WORLD, "random vectors used!\n");
		}else{
			ierr = PetscPrintf(PETSC_COMM_WORLD, "vectors provided by lapack!\n");
		}

		ierr = PetscPrintf(PETSC_COMM_WORLD,"nev: %d, ncv: %d, mpd: %d \n", nev, ncv, mpd);		
	}else{
		ierr = PetscFOpen(PETSC_COMM_WORLD, timingFile, "a", &fp);
		switch ( method ){
			case EPS_PRIMME_GD_PLUSK:
				PetscFPrintf(PETSC_COMM_WORLD,fp,"GD+K;");
				break;
			case EPS_PRIMME_JDQMR:
				PetscFPrintf(PETSC_COMM_WORLD,fp,"JDQMR;");
				break;
			default:
				PetscFPrintf(PETSC_COMM_WORLD,fp,"OTHER;");
				break;
		}
		if ( isRandom ){
			PetscFPrintf(PETSC_COMM_WORLD,fp,"RANDOM;");
		}else{
			PetscFPrintf(PETSC_COMM_WORLD,fp,"LAPACK;");
		}
		PetscFPrintf(PETSC_COMM_WORLD,fp,"%f;",t2-t1); // timing
		PetscFPrintf(PETSC_COMM_WORLD,fp,"%d;",blockSize); // blockSize
		PetscFPrintf(PETSC_COMM_WORLD,fp,"%d;",nev); // nev
		PetscFPrintf(PETSC_COMM_WORLD,fp,"%d;",ncv); // ncv
		
		PetscFPrintf(PETSC_COMM_WORLD,fp,"\n");
		
		PetscFClose(PETSC_COMM_WORLD, fp);		
	}
	
	ierr = PetscFree(scalars);
	ierr = PetscFree(idxm);
	ierr = PetscFree(idxn);
	
	EPSDestroy(&eps);
	MatDestroy(&matrix);
	ierr = SlepcFinalize();CHKERRQ(ierr);
	return 0;
}
