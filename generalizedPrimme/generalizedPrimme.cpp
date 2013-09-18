//
//  generalizedPrimme
//
//  Created by Martin Perdacher on 2013-09-13.
//  Copyright (c) 2013 . All rights reserved.
//
// example call:
// ./generalizedPrimme -hmat CaFe2As2.1.2.hmat.dat -smat CaFe2As2.1.2.smat.dat -nev 136
// ./generalizedPrimme -hmat CaFe2As2.1.2.hmat.double.dat -smat CaFe2As2.1.2.smat.double.dat -nev 136
static char help[] = "reducing the generalized linear eigenproblem to a standard problem and solve it with primme\n";

#include "slepceps.h"
#include <iostream>

#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
	Mat hmat,smat;
	Vec xr,xi,eigenvalues; // xr real, xi imaginary, eigenvalues
	PetscInt n=0, nconv,nev=0;
	PetscScalar number, kr, ki;	
	PetscReal re,im,error;
	PetscViewer viewer; 
	PetscBool flag;
	EPSType type;
	EPS eps;
	int ierr,i;
	char hmatfilename[PETSC_MAX_PATH_LEN]="", smatfilename[PETSC_MAX_PATH_LEN]="", 
		eigvecfilename[PETSC_MAX_PATH_LEN]="", eigvalfilename[PETSC_MAX_PATH_LEN]="";
	
	SlepcInitialize(&argc,&argv,(char*)0,help);
	
	ierr = PetscOptionsGetString(PETSC_NULL,"-hmat",hmatfilename,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetString(PETSC_NULL,"-smat",smatfilename,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetString(PETSC_NULL,"-rvec",eigvecfilename,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetString(PETSC_NULL,"-rval",eigvalfilename,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);	
	ierr = PetscOptionsGetInt(PETSC_NULL,"-nev",&nev,PETSC_NULL);CHKERRQ(ierr);
	
	if ( strcmp(hmatfilename, "") == 0 ||
		 strcmp(smatfilename, "") == 0 ||
//		 strcmp(eigvecfilename, "") == 0 ||
//		 strcmp(eigvalfilename, "") == 0 ||		
		 nev == 0 ){
		SETERRQ(PETSC_COMM_WORLD,1,"Error in parameterlist.\nExample call: ./generalized -hmat matrixA.dat -smat matrixB.dat -nev 136 -wvec eigenvectors.dat  ");			
	}	
	
// loading matrix hmat (a)
	ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,hmatfilename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
	ierr = MatCreate(PETSC_COMM_WORLD,&hmat);CHKERRQ(ierr);
	ierr = MatSetFromOptions(hmat);CHKERRQ(ierr);
	ierr = MatLoad(hmat,viewer);CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

// loading matrix smat (b)
	ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,smatfilename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
	ierr = MatCreate(PETSC_COMM_WORLD,&smat);CHKERRQ(ierr);
	ierr = MatSetFromOptions(smat);CHKERRQ(ierr);
	ierr = MatLoad(smat,viewer);CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);	

	ierr = MatGetVecs(hmat,NULL,&xr);CHKERRQ(ierr);
	ierr = MatGetVecs(hmat,NULL,&xi);CHKERRQ(ierr);
	ierr = MatGetVecs(hmat,NULL,&eigenvalues);CHKERRQ(ierr);

//	int j=0;
//	PetscScalar value1[25];
//	PetscInt idxn[5]={0,1,2,3,4}, idxm[5]={0,1,2,3,4},row,col;
//
//	MatGetValues(hmat, 5, idxn, 5, idxm, value1);
//	
//	for ( i= 0 ; i < 5 ; ++i ){
//		for (  j = 0 ; j < 5 ; ++j ){
//			PetscPrintf(PETSC_COMM_WORLD,"(%g \t %g), ", PetscRealPart(value1[i*5+j]),PetscImaginaryPart(value1[i*5+ j]));CHKERRQ(ierr);			
//		}
//		PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);			
//	}

// set up and solve eigensystem
	ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);
	ierr = EPSSetOperators(eps,hmat,NULL);CHKERRQ(ierr);
	ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr);
	ierr = EPSSetDimensions(eps, nev, PETSC_DECIDE, PETSC_DECIDE ); //!! does not really work, always all eigenvectors are computed !!
	ierr = EPSIsGeneralized(eps,&flag);CHKERRQ(ierr);
	
	if ( flag ){
		ierr = PetscPrintf(PETSC_COMM_WORLD,"generalized problem.\n");CHKERRQ(ierr);
	}else{
		ierr = PetscPrintf(PETSC_COMM_WORLD,"no generalized problem.\n");CHKERRQ(ierr);
	}
	
	ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
	ierr = EPSSetType(eps, EPSPRIMME);
	ierr = EPSPRIMMESetMethod(eps, EPS_PRIMME_RQI);
	ierr = EPSSetConvergenceTest(eps,EPS_CONV_ABS);
	ierr = EPSSetFromOptions(eps);CHKERRQ(ierr); // last before eps-solve, so user can override parameters

	ierr = EPSSetUp(eps);
	ierr = EPSView(eps, PETSC_VIEWER_STDOUT_WORLD);	

	ierr = EPSGetType(eps,&type);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"using %s method to compute the solution\n", type);CHKERRQ(ierr);
	
	ierr = EPSSolve(eps);CHKERRQ(ierr);
	
//	ierr = EPSPrintSolution(eps,NULL);CHKERRQ(ierr);
	ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of converged eigenpairs:%d\n", nconv);CHKERRQ(ierr);
	
	ierr = PetscPrintf(PETSC_COMM_WORLD,"writing eigenvectors to file:\n", nconv);CHKERRQ(ierr);

// write eigenvectors to binary file		
	ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, eigvecfilename, FILE_MODE_WRITE, &viewer);CHKERRQ(ierr);
		
    for (i=0;i<nconv;i++) {
		ierr = EPSGetEigenpair(eps,i,&kr,&ki,xr,xi);CHKERRQ(ierr);
		ierr = VecSetValue(eigenvalues, i, kr, INSERT_VALUES );
    	ierr = VecView(xr,viewer);CHKERRQ(ierr); // real
    	ierr = VecView(xi,viewer);CHKERRQ(ierr); // imaginary

		ierr = EPSComputeRelativeError(eps,i,&error);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX) 
		re = PetscRealPart(kr);
		im = PetscImaginaryPart(kr);
#else
		re = kr;
		im = ki; 
#endif
		ierr = PetscPrintf(PETSC_COMM_WORLD,"%d. eigenvalue: %9F%+9F rel. error: %12G\n",i,re,im,error);CHKERRQ(ierr);

    }	
	ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);


// write eigenvalues to binary file
	ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,eigvalfilename,FILE_MODE_WRITE, &viewer);CHKERRQ(ierr);
	ierr = VecView(eigenvalues, viewer);CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
	
// clean up

	ierr = EPSDestroy(&eps);CHKERRQ(ierr);
	ierr = MatDestroy(&hmat);CHKERRQ(ierr);
	ierr = MatDestroy(&smat);CHKERRQ(ierr);
	ierr = VecDestroy(&xr);CHKERRQ(ierr);
	ierr = VecDestroy(&xi);CHKERRQ(ierr);
	
	ierr = SlepcFinalize();CHKERRQ(ierr);
	return 0;
}

