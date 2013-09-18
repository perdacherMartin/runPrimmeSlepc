//
//  inspectBinary
//
//  Created by Martin Perdacher on 2013-09-18.
//  Copyright (c) 2013 . All rights reserved.
//

static char help[] = "Simple inspection programm for petsc binary hermitian files.";

#include "slepceps.h"

#define DIM 5

#undef __FUNCT__
#define __FUNCT__ "main"
int main( int argc, char **argv )
{
	int ierr,i,j;
	char infilename[PETSC_MAX_PATH_LEN]="", temporary[PETSC_MAX_PATH_LEN]="", path[PETSC_MAX_PATH_LEN]="";
	PetscViewer viewer;
	PetscBool isTail=PETSC_FALSE,isRight=PETSC_FALSE;
	Mat mat;
	PetscInt idxn[DIM], idxm[DIM], m, n, windowX=0, windowY=0;
	PetscScalar values[DIM*DIM];
	
	SlepcInitialize(&argc,&argv,(char*)0,help);
	ierr = PetscOptionsGetString(PETSC_NULL,"-file",infilename,PETSC_MAX_PATH_LEN-1,PETSC_NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetBool(PETSC_NULL, "-tail",  &isTail, NULL);
	ierr = PetscOptionsGetBool(PETSC_NULL, "-right",  &isRight, NULL);
	ierr = PetscGetWorkingDirectory(path,PETSC_MAX_PATH_LEN);
	
	// concat filename
	strcat(temporary, path); strcat(temporary, "/"); strcat(temporary, infilename); strcpy(infilename, temporary);
	
	if ( strcmp(infilename, "") == 0  ){
		SETERRQ(PETSC_COMM_WORLD,1,"Specifiy the filename!\nExample call: ./inspectBinary -file matrix.dat");			
	}
	
	ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,infilename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
	ierr = MatCreate(PETSC_COMM_WORLD,&mat);CHKERRQ(ierr);
	ierr = MatSetFromOptions(mat);CHKERRQ(ierr);
	ierr = MatLoad(mat,viewer);CHKERRQ(ierr);
	ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
	
	MatGetSize(mat,&m,&n);
	
	if ( isTail == PETSC_TRUE ){
		windowY = m - DIM;
	}
	
	if ( isRight == PETSC_TRUE ){
		windowX = n - DIM;
	}
	
	for ( i = 0 ; i < DIM ; ++i ){
		idxm[i] = i + windowY; idxn[i] = i + windowX;
	}
	
	MatGetValues(mat,DIM,idxm,DIM,idxn,values);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Dimension: %d rows x %d columns:\n", m, n);CHKERRQ(ierr);
	ierr = PetscPrintf(PETSC_COMM_WORLD,"row index idxm:");CHKERRQ(ierr);
	for ( i = 0 ; i < DIM ; ++i ){
		ierr = PetscPrintf(PETSC_COMM_WORLD,"%d, ", idxm[i]);CHKERRQ(ierr);
	}

	ierr = PetscPrintf(PETSC_COMM_WORLD,"\ncol index idxn:");CHKERRQ(ierr);
	for ( i = 0 ; i < DIM ; ++i ){
		ierr = PetscPrintf(PETSC_COMM_WORLD,"%d, ", idxn[i]);CHKERRQ(ierr);
	}
	ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\n\n");CHKERRQ(ierr);	
	
	for ( i = 0 ; i < DIM ; ++i ){
		for ( j = 0 ; j <  DIM ; ++j ){
			ierr = PetscPrintf(PETSC_COMM_WORLD,"(%2.10e/ %2.10e) ", PetscRealPart(values[i*DIM + j]), PetscImaginaryPart(values[i*DIM + j]) );CHKERRQ(ierr);
		}
		ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);		
	}
	
	ierr = SlepcFinalize();CHKERRQ(ierr);
	return 0;
}

