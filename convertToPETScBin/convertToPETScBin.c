//
//  generalized
//
//  reducing the generalized linear eigenproblem to a standard problem and store the 
//  eigenpairs into a file
//  Created by Martin Perdacher on 2013-09-11.
//  Copyright (c) 2013 . All rights reserved.
//

static char help[] = "converter for internal matrix format to petsc binary format\n";

#include "slepceps.h"
#include <iostream>

#define PETSC_DESIRE_COMPLEX

#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
	Mat hmat,smat;
	PetscInt n=0,row,col, nev;
	PetscScalar number;	
	PetscViewer view;
	PetscBool flag;
	int ierr,i;
	char path[PETSC_MAX_PATH_LEN]="", temporary[PETSC_MAX_PATH_LEN]="";
	char infilename[PETSC_MAX_PATH_LEN]="", outfilename[PETSC_MAX_PATH_LEN]="";
	
	SlepcInitialize(&argc,&argv,(char*)0,help);
	
	ierr = PetscOptionsGetString(PETSC_NULL,"-in",infilename,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetString(PETSC_NULL,"-out",outfilename,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
	
	ierr = PetscGetWorkingDirectory(path,PETSC_MAX_PATH_LEN);
	
	if ( strcmp(infilename, "") == 0 ||
		 strcmp(outfilename, "") == 0 ||
		 n == 0 ){
		SETERRQ(PETSC_COMM_WORLD,1,"Error in parameterlist.\nExample call: ./convertToPETScBin -in matrix.hmat -out matrix.dat -n 1912");			
	}
	
//	number = 2.0 + 3.0 * PETSC_i;	
//	PetscPrintf(PETSC_COMM_WORLD,"number: %g i%g\n", number);
//  std::cout << number
	PetscPrintf(PETSC_COMM_WORLD,"processing %s into %s\n", infilename, outfilename);
	
	ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,n,n,n,PETSC_NULL,&hmat);

	FILE *file;
	double real, imag;
	PetscScalar complex;
	PetscScalar myreal;
	
	strcat(temporary, path); strcat(temporary, "/"); strcat(temporary, infilename); strcpy(infilename, temporary);
	
	ierr   = PetscFOpen(PETSC_COMM_SELF,infilename,"r",&file);CHKERRQ(ierr);

	while ( fscanf(file,"%d %d %le %le\n",&row,&col,(double*)&real, (double*)&imag) != EOF ){
		row = row - 1; col = col -1; // adjustment
		
		if ( row == col ){
			complex = real /*+ imag*PETSC_i*/;
			ierr = MatSetValues(hmat,1,&row,1,&col,&complex,INSERT_VALUES);CHKERRQ(ierr);
		}else{
			complex = real /*+ imag*PETSC_i */;
			ierr = MatSetValues(hmat,1,&row,1,&col,&complex,INSERT_VALUES);CHKERRQ(ierr);
			complex = real /*- imag*PETSC_i */;
			ierr = MatSetValues(hmat,1,&col,1,&row,&complex,INSERT_VALUES);CHKERRQ(ierr);			
		}
			
	}
	
	fclose(file);
	ierr = MatAssemblyBegin(hmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(hmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

	ierr = MatIsHermitian(hmat,0.00000000000001,&flag); CHKERRQ(ierr);
	if ( !flag ){
		SETERRQ(PETSC_COMM_WORLD,1,"Error. Output matrix is not hermitian.");
	}
	
// writing
	strcpy(temporary,"");
	strcat(temporary, path); strcat(temporary, "/"); strcat(temporary, outfilename); strcpy(outfilename, temporary);
	
	ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,outfilename,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
	ierr = MatView(hmat,view);CHKERRQ(ierr);
	
	ierr = PetscViewersDestroy(&view);
	
	ierr = SlepcFinalize();CHKERRQ(ierr);
	return 0;
}

