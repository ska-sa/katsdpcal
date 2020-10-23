#!groovy

@Library('katsdpjenkins') _
katsdp.killOldJobs()

katsdp.setDependencies([
    'ska-sa/katsdpdockerbase/new-rdma-core',
    'ska-sa/katpoint/master',
    'ska-sa/katdal/master',
    'ska-sa/katsdpsigproc/master',
    'ska-sa/katsdpservices/master',
    'ska-sa/katsdptelstate/master'])
katsdp.standardBuild(
    python3: true,
    python2: false,
    docker_venv: true,
    push_external: true,
    katsdpdockerbase_ref: 'new-rdma-core')
katsdp.mail('sdpdev+katsdpcal@ska.ac.za')
