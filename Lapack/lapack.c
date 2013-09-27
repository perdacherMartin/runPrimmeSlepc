//
//  lapack call, to compute eigenvalues with zheevr
//
//  example call:
//  ./lapack -hmat CaFe2As2.small.1_1_hmat -smat CaFe2As2.small.1_1_smat -n 1912 -stw CaFe2As2.small.1_1.standard.dat -evecw CaFe2As2.small.1_1.lapack.evec.dat -relSlepc reativeSlepcError.csv -relLapack relativeLapackError.csv -evecSlepcw CaFe2As2.small.1_1.slepc.evec.dat -nev 136 -timing timing.csv
//
//  Created by Martin Perdacher on 2013-08-27.
//  Copyright (c) 2013 . All rights reserved.
//

#include "lapack.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv ){ 
	PetscErrorCode ierr;
	PetscViewer viewer;
	PetscBool storeRelativeLapackError=PETSC_TRUE, storeRelativeSlepcError=PETSC_TRUE;
	PetscInt n,row,col, nev,*ix;
	PetscScalar *lambdaF;
	int size,i,j;
	T_complex *Hmat=NULL,*Smat=NULL,*H=NULL;
	char hmatfilename[PETSC_MAX_PATH_LEN]="", smatfilename[PETSC_MAX_PATH_LEN]="", relErrorLapackFilename[PETSC_MAX_PATH_LEN]="",
		standardFilename[PETSC_MAX_PATH_LEN]="", evecFilename[PETSC_MAX_PATH_LEN]="", relErrorSlepcFilename[PETSC_MAX_PATH_LEN]="",
		evecSlepcFilename[PETSC_MAX_PATH_LEN]="", timingFile[PETSC_MAX_PATH_LEN]="", evalFile[PETSC_MAX_PATH_LEN]="";
	PetscReal *lambda=NULL; // the eigenvalues are stored here
	T_complex *y=NULL; // matrix containing the eigenvectors, i-th column corresponds to i-th eigenvector
	FILE *fp;
	Vec lambdaVec;
	
	SlepcInitialize(&argc,&argv,(char*)0,help);

	// no need for multiprocessors
  	ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
	if (size > 1) SETERRQ(PETSC_COMM_WORLD,1,"supports uniprocessor only!!!\n");

	ierr = PetscOptionsGetString(PETSC_NULL,"-hmat",hmatfilename,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetString(PETSC_NULL,"-smat",smatfilename,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(PETSC_NULL,"-nev",&nev,PETSC_NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetString(PETSC_NULL,"-relLapack",relErrorLapackFilename,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetString(PETSC_NULL,"-relSlepc",relErrorSlepcFilename,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetString(PETSC_NULL,"-stw",standardFilename,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetString(PETSC_NULL,"-timing",timingFile,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetString(PETSC_NULL,"-evecw",evecFilename,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetString(PETSC_NULL,"-evecSlepcw",evecSlepcFilename,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetString(PETSC_NULL,"-evalFile",evalFile,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);
	
	lambda =  (PetscReal *) malloc(n * sizeof(PetscReal)); 
	if ( lambda == NULL ) { PetscPrintf(PETSC_COMM_WORLD,"failed to allocate workspace for lambda."); exit(1); }	

	y =  (T_complex *) malloc(n * n * sizeof(T_complex)); 
	if ( y == NULL ) { PetscPrintf(PETSC_COMM_WORLD,"failed to allocate workspace for y."); exit(1);}	
		
	for ( i = 0 ; i < n*n ; ++i ){ y[i].real = 0.0; y[i].imag = 0.0; }	
		
	Hmat = ReadSequentiallyFromFile(hmatfilename, n);
	if ( ! Hmat ) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, strcat("Could not open file ", hmatfilename)); }
	
	Smat = ReadSequentiallyFromFile(smatfilename, n);
	if ( ! Smat ) { SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, strcat("Could not open file ", smatfilename)); }
	
	H = ReduceGeneralizedToStandardform(Hmat,Smat,n);
	
	// solves with fortran lapack and does a transpose of H
	ierr = SolveH(H, n, nev, lambda, y, relErrorLapackFilename, timingFile );CHKERRQ(ierr);

	// write eigenvectors to file using petsc_binary format
	// transpose == PETSC_FALSE, means vectors are stored line by line
	ierr = WriteToPetscBinary(PETSC_FALSE, y, evecFilename, n);CHKERRQ(ierr);

	// write generalized matrix to file (petsc_binary format), slepc-primme cannot handle generalized eigenproblems
	// transpose == PETSC_FALSE, means row-major format, like c
	ierr = WriteToPetscBinary(PETSC_FALSE, H, standardFilename, n);CHKERRQ(ierr);

	// solve with slepc and EPSLAPACK
	TransposeHermitian(Hmat, n);
	TransposeHermitian(Smat, n);
	// do not transpose h, because it is done within the solve solveH routine
//	ierr = SolveWithSlepc(Hmat, Smat, relErrorSlepcFilename, n, evecSlepcFilename,timingFile);CHKERRQ(ierr);
	
	// write lambda to file
	ierr = PetscMalloc(n*sizeof(PetscInt), &ix);CHKERRQ(ierr);
	ierr = PetscMalloc(n*sizeof(PetscScalar), &lambdaF);CHKERRQ(ierr);
	for ( i = 0 ; i < n ; ++i ){ ix[i] = i; lambdaF[i] = (PetscScalar)lambda[i]; }
	ierr = VecCreate(PETSC_COMM_WORLD, &lambdaVec);CHKERRQ(ierr);
	ierr = VecSetSizes(lambdaVec,n,n);CHKERRQ(ierr);
	ierr = VecSetFromOptions(lambdaVec);CHKERRQ(ierr);
	ierr = VecSetValues(lambdaVec, n, ix, lambdaF, INSERT_VALUES);CHKERRQ(ierr);
	
	ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,evalFile,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
	ierr = VecView(lambdaVec,viewer);CHKERRQ(ierr);
	
	ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
	
	if ( strcmp(timingFile, "" ) != 0 ){
		ierr = PetscFOpen(PETSC_COMM_SELF, timingFile, "a", &fp);
		fprintf(fp,"\n");
		PetscFClose(PETSC_COMM_SELF, fp);	
	}else{
		PetscPrintf(PETSC_COMM_WORLD,"\n");				
	}
	
	free(Hmat); free(Smat); free(H);
	ierr = SlepcFinalize();CHKERRQ(ierr);
  	return 0;
}

#undef __FUNCT__
#define __FUNCT__ "ReduceGeneralizedToStandardform"
T_complex* ReduceGeneralizedToStandardform(T_complex *Hmat,T_complex *Smat, PetscInt const n){
	int lda = n /* Hmat */ , ldb = n /* Smat */, myn=n ;
	int info, itype=1,i,j;
	T_complex *H,*Scopy;
	
	Scopy = (T_complex *) malloc(n * n *sizeof(T_complex));
	if ( Scopy == NULL ) { PetscPrintf(PETSC_COMM_WORLD,"failed to allocate workspace for H."); exit(1);}
	memcpy(Scopy, Smat, n*n*sizeof(T_complex));
	
	H =  (T_complex *) malloc(n * n *sizeof(T_complex)); // allocate H
	if ( H == NULL ) { PetscPrintf(PETSC_COMM_WORLD,"failed to allocate workspace for H."); exit(1);}
	memcpy(H, Hmat, n*n*sizeof(T_complex));
	
	zpotrf("Lower", &myn, Scopy, &myn, &info); // cholesky factorization

    if( info > 0 ) {
    	PetscPrintf(PETSC_COMM_WORLD,"Cholesky factorization failed to compute! The leading minor of order %d is not positive definite\n", info);
		exit( 1 );
    }
	
	memcpy(H, Hmat, n*n*sizeof(T_complex));
	zhegst(&itype, "Lower", &myn, H, &lda, Scopy, &ldb, &info);
	// zhegest solves for the lower part of H, stored in column-major format
	// set the upper part of H (column-major format)
	for ( i = 0 ; i < n ; ++i ){
		for ( j = i + 1 ; j < n ; ++j ){
			H[j*n + i].real = H[i*n + j ].real; 
			H[j*n + i].imag = -H[i*n + j].imag;
		}
	}

//	for ( i = 0 ; i < 5 ; ++i ){
//		for ( j = 0 ; j < 5 ; ++j ){
//			PetscPrintf(PETSC_COMM_WORLD,"(%2.10e : %2.10e)", H[i*n + j ].real, H[i*n + j].imag);
//		}
//		PetscPrintf(PETSC_COMM_WORLD,"\n");
//	}
	
    if( info > 0 ) {
    	PetscPrintf(PETSC_COMM_WORLD,"Failed to reduce to standard form.\n");
		exit( 1 );
    }	

	free(Scopy);
	return H;
}

#undef __FUNCT__
#define __FUNCT__ "GetTimeMinimum"
PetscErrorCode GetTimeMinimum(PetscLogDouble *times, PetscLogDouble &minimum, char* const timeFilename){
	PetscErrorCode ierr=0;
	FILE *file;
	int i;
	
	minimum = times[0];
	for ( i=0 ; i < REPETITIONS ; ++i ){
		if ( times[i] < minimum ){
			minimum = times[i];
		}
	}
	
	if ( strcmp(timeFilename, "" ) != 0 ){
		ierr = PetscFOpen(PETSC_COMM_SELF, timeFilename, "a", &file);
		fprintf(file,"%f;", minimum);
		PetscFClose(PETSC_COMM_SELF, file);	
	}else{
		printf("%f;", minimum);				
	}	
	return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "LapackAll"
PetscErrorCode LapackAll(T_complex *H, PetscInt const n, char* const timeFilename, PetscReal *lambda, T_complex *y){
	PetscErrorCode ierr;
	FILE *file;
	int N = n, lda = N, ldz = lda,i,j,k;
	T_complex *work=NULL,wkopt;	
	T_complex *Hcopy=NULL;
	double rwork[3*n-2]; // for zheev
	int *iwork,lwork=-1,info;	
	PetscLogDouble times[REPETITIONS],t1,t2,minimum;
		
	// copy, because lapack destroys the matrix 
	Hcopy = (T_complex *) malloc(n * n *sizeof(T_complex));

	// query workspace
	zheev( "Vectors", "Lower", &N, Hcopy, &lda, lambda, &wkopt, &lwork, rwork, &info );

	lwork = (int)wkopt.real;
    work = (T_complex*)malloc( lwork*sizeof(T_complex) );

	for ( i = 0 ; i < REPETITIONS ; ++i ){
		memcpy(Hcopy, H, n*n*sizeof(T_complex));

		ierr = PetscTime(&t1);CHKERRQ(ierr);
		zheev( "Vectors", "Lower", &N, Hcopy, &lda, lambda, work, &lwork, rwork, &info );
		ierr = PetscTime(&t2);CHKERRQ(ierr);
		
		times[i] = t2-t1;
		memcpy(y, Hcopy, n*n*sizeof(T_complex)); // eigenvectors are stored in Hcopy, move them to y		

	}
	
	ierr = GetTimeMinimum(times, minimum, timeFilename);CHKERRQ(ierr);
	
	free(work);
	free(Hcopy);
	return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "LapackRobust"
PetscErrorCode LapackRobust(T_complex *H, PetscInt const n, char* const timeFilename, PetscInt const nev, PetscReal *lambda, T_complex *y){
	PetscErrorCode ierr=0;
	FILE *file;
	PetscLogDouble times[REPETITIONS],t1,t2,minimum;
	int N = n, lda = N, ldz = lda,i,j,k;
	int isuppz[n], lwork, lrwork, liwork, info;
	int il,iu,m,iwkopt;
	double vl, vu, abstol, rwkopt;	
	T_complex *work=NULL,wkopt;	
	T_complex *Hcopy=NULL;
	double *rwork; 
	int *iwork;

	// half-opend intervall, vl < vu
	vl = -5.0; vu = 5.0; // will not have an effect, set to Interval
	// if range (2nd paramter) = I, calculating il-th eigenvalue until il-th eigenvalue
	il = 1 ; iu = nev;
	
	// abstol = precision, when negative using default precision
	abstol = -1.0;
	// query and allocate the optimal workspace 
	lwork = -1;
    lrwork = -1;
    liwork = -1;
	Hcopy = (T_complex *) malloc(n * n *sizeof(T_complex));
	
	// query workspace
    zheevr( "Vectors", "Interval", "Lower", &N, Hcopy, &lda, &vl, &vu, &il, &iu,
                    &abstol, &m, lambda, y, &ldz, isuppz, &wkopt, &lwork, &rwkopt, &lrwork,
                    &iwkopt, &liwork, &info );	

	// allocate workspace
	lwork = (int)wkopt.real;
	work = (T_complex*)malloc( lwork*sizeof(T_complex) );
	if ( work == NULL ) { PetscPrintf(PETSC_COMM_WORLD,"failed to allocate workspace for work."); exit(1);}
	
	lrwork = (int)rwkopt;
	rwork = (double*)malloc( lrwork*sizeof(double) );
	if ( rwork == NULL ) { PetscPrintf(PETSC_COMM_WORLD,"failed to allocate workspace for work."); exit(1);}
	
	liwork = iwkopt;	
	iwork = (int*)malloc( liwork*sizeof(int) );
	if ( rwork == NULL ) { PetscPrintf(PETSC_COMM_WORLD,"failed to allocate workspace for work."); exit(1);}	
	
	// lapack call to compute the standard eigenvalue problem
	for ( i = 0 ; i < REPETITIONS ; ++i ){
		memcpy(Hcopy, H, n*n*sizeof(T_complex));

		ierr = PetscTime(&t1);CHKERRQ(ierr);
		zheevr( "Vectors", "Interval", "Lower", &N, Hcopy, &lda, &vl, &vu, &il, &iu,
	                    &abstol, &m, lambda, y, &ldz, isuppz, work, &lwork, rwork, &lrwork,
	                    iwork, &liwork, &info );
		ierr = PetscTime(&t2);CHKERRQ(ierr);
		times[i] = t2-t1;
		memcpy(y, Hcopy, n*n*sizeof(T_complex)); // eigenvectors are stored in Hcopy, move them to y		
	}

	if( info > 0 ) { 
		ierr = PetscPrintf(PETSC_COMM_WORLD,"The algorithm failed to compute eigenvalues.\n");
		ierr = PetscPrintf(PETSC_COMM_WORLD,"The %d-th eigenvector failed to converge.\n",info);
		exit( 1 );
	}
	
	if( info < 0 ) {
		ierr = PetscPrintf(PETSC_COMM_WORLD,"The algorithm failed to compute eigenvalues.\n");
		ierr = PetscPrintf(PETSC_COMM_WORLD,"The %d-th argument had an illegal value.\n",(-info) );
		exit( 1 );
	}

	ierr = GetTimeMinimum(times, minimum, timeFilename);CHKERRQ(ierr);	
	
	free(work);
	free(rwork);
	free(iwork);	
	free(Hcopy);
	
	return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "SolveH"
// 	solves with fortran lapack and does a hermitian transpose of H
PetscErrorCode SolveH(T_complex *H, PetscInt const n, PetscInt const nev, PetscReal *lambda, T_complex *y, char* const relErrorLapackFilename, char* const timeFilename){
	PetscErrorCode ierr=0;
	FILE *file;
	double vl, vu, abstol, rwkopt;	
	T_complex *work=NULL,wkopt;	
	T_complex *Hcopy=NULL;
	double rwork[3*n-2]; // for zheev
	int i,j,k,evecCount;
	
	LapackRobust(H, n, timeFilename, nev, lambda, y); evecCount = nev;
	LapackAll(H, n, timeFilename, lambda, y); evecCount = n;	

	// for matrix vector multiplication transpose the matrix back, because we are calling cblas
	TransposeHermitian(H, n);

	T_complex alphaLambda, alphaOne; alphaOne.real = 1.0; alphaOne.imag = 0.0;
	T_complex betaZero; betaZero.real = 0.0; betaZero.imag = 0.0;
	T_complex *residual=NULL, *lambdaidentity=NULL, tempResidual;
	int inc=1, incN=n;
		
	// calculating a residual | H * y - lambda * y | using blas routines		
	if ( strcmp(relErrorLapackFilename,"") != 0 ) {
		
		ierr = PetscFOpen(PETSC_COMM_SELF, relErrorLapackFilename, "w", &file);
		
		// allocate and initialize vector to calculate residual
		residual = (T_complex*) malloc( n*sizeof(T_complex) );
		
		i=0;
		double sum=0.0;
		T_complex y2[n];
		
		for ( i = 0 ; i < evecCount ; ++i ){ 
			for ( j = 0 ; j < n ; ++j ) {
				// betaZero is zero, so residual need to be set to zero
				residual[j].real = 0.0; residual[j].real = 0.0;
			}
			
			// residual = alphaOne * H * y(i) + beta (=0) * residual 		
			cblas_zgemv(CblasColMajor, CblasTrans, n, n, &alphaOne, H, n, y + i*n, 1, &betaZero, residual, 1); 
			alphaLambda.real = -1.0 * lambda[i]; alphaLambda.imag = 0.0 ;
			cblas_zaxpy(n, &alphaLambda, y + i*n, 1, residual, 1 ); // residual = residual + ( -lambda(i) * y(i) )
			
			for ( k = 0 ; k < n ; ++k ){ // residual = residual ^ 2
				tempResidual.real = residual[k].real * residual[k].real - residual[k].imag * residual[k].imag;
				tempResidual.imag = residual[k].real * residual[k].imag + residual[k].imag * residual[k].real;
				
				residual[k].real = tempResidual.real; residual[k].imag = tempResidual.imag;
			}
			
			fprintf(file,"%d;%1.10e\n",i+1, sqrt(cblas_dzasum(n, residual, 1)));
		}
		
		PetscFClose(PETSC_COMM_SELF, file);		
	}
	
	return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "ReadSequentiallyFromFile"
T_complex* ReadSequentiallyFromFile(char const *filein, PetscInt const n){
	PetscInt row,col;
	double real,imag;
	char strReal[PETSC_MAX_PATH_LEN], strImag[PETSC_MAX_PATH_LEN];
	PetscErrorCode ierr;
	T_complex *matrix = NULL;
	int i=0;
	
	matrix = (T_complex *)malloc(n * n *sizeof(T_complex)); // initialize n*n array
		if ( matrix == NULL ) { PetscPrintf(PETSC_COMM_WORLD,"failed to allocate workspace for rwork."); exit(1);}
	FILE* file;
	
	ierr = PetscFOpen(PETSC_COMM_SELF, filein, "r", &file);
	
	if (!file) {
		return NULL;
	}

	// read the file in (sequentially), only the lower triangular part of the hermitian is read in
	while ( fscanf(file,"%d %d %le %le\n",&row,&col,(double*)&real, (double*)&imag) != EOF ){
		i++;
		row = row - 1; col = col -1; // adjustment
		matrix[row*n + col].real = real; matrix[row*n + col].imag = imag;  
		if ( row != col ){
			matrix[col*n + row].real = real; matrix[col*n + row].imag = -imag; 			
		}		
	}
	
	PetscFClose(PETSC_COMM_SELF, file);
	
	//transpose Hcopy, because calling fortran routines...
	TransposeHermitian(matrix, n);
		
	return matrix;
}

#undef __FUNCT__
#define __FUNCT__ "TransposeHermitian"
// adjungate transpose for hermitian matrix
PetscErrorCode TransposeHermitian(T_complex *matrix, PetscInt const n){
	int i,j;
	PetscErrorCode ierr = 0;
	
	for ( i = 0 ; i < n ; ++i ){
		for ( j = i+1 ; j < n ; ++j ){
			matrix[i*n+j].imag = -matrix[i*n+j].imag; matrix[j*n+i].imag = -matrix[j*n+i].imag;
		}
	}
	return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "WriteToPetscBinary"
PetscErrorCode WriteToPetscBinary(PetscBool const doTranspose, T_complex *cmatrix, char* const filename, PetscInt const n){
	PetscViewer viewer;
	PetscErrorCode ierr=0;
	Mat matrix;
	int i,j;
	PetscScalar complex;
	PetscScalar imag;
		
	ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,n,n,n,PETSC_NULL,&matrix);CHKERRQ(ierr);
		
	for ( i = 0 ; i <  n ; ++i ){
		for ( j = 0 ; j < n ; ++j ){
			
			if ( doTranspose == PETSC_TRUE ){
				complex = cmatrix[j*n +i].real + cmatrix[j*n +i].imag * PETSC_i;	
			}else{
				complex = cmatrix[i*n +j].real + cmatrix[i*n +j].imag * PETSC_i;
			}
			ierr = MatSetValues(matrix,1,&i,1,&j,&complex,INSERT_VALUES);CHKERRQ(ierr);
		}
	}
	ierr = MatAssemblyBegin(matrix,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(matrix,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);


	ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
	ierr = MatView(matrix,viewer);CHKERRQ(ierr);
	
	ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);	
	
	return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "SolveWithSlepc"
PetscErrorCode SolveWithSlepc(T_complex *Hmat, T_complex *Smat, char* const relErrorSlepcFilename, PetscInt const n, char* const evecSlepcFilename, char* const timeFilename){
	
	Mat H,S,eigenvec;
	EPSType type;
	PetscLogDouble times[REPETITIONS],t1,t2,minimum;
	EPS eps;
	FILE *file;
	PetscBool writeToFile=PETSC_TRUE;
	Vec xr, xi;
	PetscScalar xreal[n], ximag[n], ki, kr;
	PetscReal error;
	PetscInt ix[n], nconv;
	T_complex *cEigenvec=NULL;
	int i,j;
	PetscErrorCode ierr;
	
	for ( i = 0 ; i < n ; ++i ){ ix[i] = i; } // needed for vecgetvalues
	
	cEigenvec =  (T_complex *) malloc(n * n * sizeof(T_complex)); 
	if ( cEigenvec == NULL ) { PetscPrintf(PETSC_COMM_WORLD,"failed to allocate workspace for cEigenvec."); exit(1); }	
	
	VecCreateSeq(PETSC_COMM_SELF, n, &xr);
	VecDuplicate(xr,&xi);
	
	if ( strcmp(evecSlepcFilename,"") == 0 ) {
		writeToFile=PETSC_FALSE;
	}
	
	ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,n,n,n,PETSC_NULL,&H);CHKERRQ(ierr);
	ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,n,n,n,PETSC_NULL,&S);CHKERRQ(ierr);
		
	ierr = SetUpPetscMatrix(Hmat, H, n);CHKERRQ(ierr);
	ierr = SetUpPetscMatrix(Smat, S, n);CHKERRQ(ierr);

	for ( i = 0 ; i < REPETITIONS ; ++i ){
		if ( i !=  0 ){
			EPSDestroy(&eps);
		}
		ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);
		ierr = EPSSetOperators(eps,H,S);CHKERRQ(ierr);
		//ierr = EPSSetOperators(eps,H,NULL);CHKERRQ(ierr);
		ierr = EPSSetProblemType(eps,EPS_GHEP);CHKERRQ(ierr);
		ierr = EPSSetType(eps, EPSLAPACK);
		// ierr = EPSSetDimensions(eps, nev, PETSC_DECIDE, PETSC_DECIDE );  does not really work with epslapack
		ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
		ierr = PetscTime(&t1);CHKERRQ(ierr);
		ierr = EPSSolve(eps);CHKERRQ(ierr);
		ierr = PetscTime(&t2);CHKERRQ(ierr);
		times[i] = t2 -t1;
	}

	ierr = GetTimeMinimum(times, minimum, timeFilename);CHKERRQ(ierr);		
	
	ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
	
	if ( writeToFile == PETSC_TRUE ) {
		ierr = PetscFOpen(PETSC_COMM_SELF, relErrorSlepcFilename, "w", &file);
	}
	
	for (i=0;i<nconv;i++) {
		ierr = EPSGetEigenpair(eps,i,&kr,&ki,xr,xi);CHKERRQ(ierr);
		
		ierr = VecGetValues(xr, n, ix, xreal); CHKERRQ(ierr);
		ierr = VecGetValues(xi, n, ix, ximag); CHKERRQ(ierr); // is always zero with complex c version of slepc
		
//		if ( i < 20 ){
//			PetscPrintf(PETSC_COMM_WORLD,"%d (%2.10e)\n", i, kr);
//		}
		
		for ( j = 0 ; j < n ; ++j ){
			cEigenvec[i*n + j].real = PetscRealPart(xreal[j]);
			cEigenvec[i*n + j].imag = PetscImaginaryPart(xreal[j]);
		}

		if ( writeToFile == PETSC_TRUE ) {
			ierr = EPSComputeRelativeError(eps,i,&error);CHKERRQ(ierr);
			fprintf(file,"%d;%1.10e\n",i+1, error);
		}
	}
	
	if ( writeToFile == PETSC_TRUE ) {
		PetscFClose(PETSC_COMM_SELF, file);
	}	
	
	if ( strcmp(evecSlepcFilename,"") != 0 ) {
		WriteToPetscBinary(PETSC_FALSE, cEigenvec, evecSlepcFilename, n);
	}
	free(cEigenvec);
	
	return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "SetUpPetscMatrix"
PetscErrorCode SetUpPetscMatrix(T_complex *cmatrix, Mat matrix, PetscInt const n){
	int i,j;
	PetscScalar complex; 
	PetscErrorCode ierr;
	
	for ( i = 0 ; i < n ; ++i ){
		for ( j = 0 ; j < n ; ++j ){
			complex = cmatrix[i*n + j].real + cmatrix[i*n + j].imag * PETSC_i;
			ierr = MatSetValues(matrix,1,&i,1,&j,&complex,INSERT_VALUES);CHKERRQ(ierr);
		}
	}
	
	ierr = MatAssemblyBegin(matrix,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(matrix,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

	return ierr;
}


