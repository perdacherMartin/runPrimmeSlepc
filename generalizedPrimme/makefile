#
#  makefile for generalizedPrimme
#
#  Created by Martin Perdacher on 2013-09-13.
#  Copyright (c) 2013 . All rights reserved.
#

generalizedPrimme: generalizedPrimme.o chkopts
	-${CLINKER} -g -debug -o generalizedPrimme generalizedPrimme.o ${SLEPC_LIB}
	${RM} generalizedPrimme.o

include ${SLEPC_DIR}/conf/slepc_common
