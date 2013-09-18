#
#  makefile for calling lapack routine in slepc
#
#  Created by Martin Perdacher on 2013-08-27.
#  Copyright (c) 2013 . All rights reserved.
#
ALL: lapack

CFLAGS = -I/usr/local/intel/mkl/10.2.5.035/include

lapack: lapack.o chkopts
	-${CLINKER} -o lapack lapack.o ${SLEPC_LIB} 
	${RM} lapack.o

include ${SLEPC_DIR}/conf/slepc_common
