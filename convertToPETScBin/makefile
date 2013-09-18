#
#  makefile
#
#  Created by Martin Perdacher on 2013-09-11.
#  Copyright (c) 2013 . All rights reserved.
#

convertToPETScBin: convertToPETScBin.o chkopts
	-${CLINKER} -o convertToPETScBin convertToPETScBin.o ${SLEPC_LIB}
	${RM} convertToPETScBin.o

include ${SLEPC_DIR}/conf/slepc_common
