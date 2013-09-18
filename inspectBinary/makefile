#
#  makefile
#
#  Created by Martin Perdacher on 2013-09-18.
#  Copyright (c) 2013 . All rights reserved.
#

inspectBinary: inspectBinary.o chkopts
	-${CLINKER} -o inspectBinary inspectBinary.o ${SLEPC_LIB}
	${RM} inspectBinary.o

include ${SLEPC_DIR}/conf/slepc_common
