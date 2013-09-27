#!/bin/bash

NODES=1
OUTPUTDIR=driverout
LOGDIR=logs
METHODS="jdqmr gd_plusk"
EPS_NEV=25
STDDIR=std
EVECGUESSDIR=lapack
# ?? be careful, can be crushed, if two results write exaclty in the same time ??
TIMINGFILE=timing.csv

mkdir -p $OUTPUTDIR
mkdir -p $OUTPUTDIR/$LOGDIR

for METHOD in $METHODS 
do
	# 1..29 for 30 matrices in the sequences
	for NUMBER in {1..29}
	do
		#INC is equal to l in the sequence
		INC=$(expr $NUMBER + 1)
		STDFILE=$(printf 'CaFe2As2.small.standard.%03d%03d.dat' 1 $INC)
		EVECFILE=$(printf 'CaFe2As2.small.lapack.%03d%03d.dat' 1 $NUMBER)
		
		EXECUTIONFILE="callDriverSub.$METHOD.$NUMBER.sh"
		
		echo "#!/bin/bash" > $EXECUTIONFILE
		echo "#MSUB -l nodes=${NODES}:ppn=8" >> $EXECUTIONFILE
		echo "#MSUB -l walltime=4:0:00" >> $EXECUTIONFILE
		echo "#MSUB -e $OUTPUTDIR/$LOGDIR/calldriver.err.$METHOD.$NUMBER.txt" >> $EXECUTIONFILE
		echo "#MSUB -o $OUTPUTDIR/$LOGDIR/calldriver.out.$METHOD.$NUMBER.txt" >> $EXECUTIONFILE
		echo "" >> $EXECUTIONFILE
		echo "SAVEIFS=\$IFS" >> $EXECUTIONFILE
		echo "IFS=\$(echo -en \"\\n\")" >> $EXECUTIONFILE
		echo "" >> $EXECUTIONFILE		
		echo "export OMP_NUM_THREADS=8" >> $EXECUTIONFILE
		echo -n "mpiexec -np ${NODES} driver -stdFile $STDDIR/${STDFILE} -evecGuess $EVECGUESSDIR/$EVECFILE -eps_nev $EPS_NEV " >> $EXECUTIONFILE
		echo " -eps_primme_method $METHOD -timingFile $OUTPUTDIR/$TIMINGFILE " >> $EXECUTIONFILE
		echo "" >> $EXECUTIONFILE
		echo -n "mpiexec -np ${NODES} driver -stdFile $STDDIR/${STDFILE} -evecGuess $EVECGUESSDIR/$EVECFILE -eps_nev $EPS_NEV " >> $EXECUTIONFILE
		echo " -eps_primme_method $METHOD -timingFile $OUTPUTDIR/$TIMINGFILE -random" >> $EXECUTIONFILE
		echo "" >> $EXECUTIONFILE
		echo "\$IFS=\$SAVEIFS" >> $EXECUTIONFILE
		chmod 755 $EXECUTIONFILE
		msub $EXECUTIONFILE
		sleep 2
	done
done
