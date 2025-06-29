/Users/mwardynski/Documents/ds/_semestr_8/pp
/net/afscra/people/plgmwardynski -- Ares
/net/tscratch/people/plgmwardynski -- Athena

/net/tscratch/people/plgmwardynski/ADNI_derivatives/

:%s#/old/path#/new/path#g
:%s#/Users/mwardynski/Documents/ds/_semestr_8/pp#/net/tscratch/people/plgmwardynski#g

python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt

    hpc-grants - shows available grants, resource allocations, consumed resourced
    hpc-fs - shows available storage
    hpc-jobs - shows currently pending/running jobs
    hpc-jobs-history - shows information about past jobs

sbatch job.slurm