#!/usr/bin/env groovy

node {
  def govuk = load('/var/lib/jenkins/groovy_scripts/govuk_jenkinslib.groovy')
  def repoName = JOB_NAME.split('/')[0]

  stage('Checkout') {
        govuk.checkoutFromGitHubWithSSH(repoName)
  }

  stage('Merge the master branch') {
    govuk.mergeMasterBranch()
  }

  stage('Setup virtualenv') {
    sh("rm -rf ./venv")
    sh("virtualenv --python=python3 --no-site-packages ./venv")
    govuk.setEnvar("VIRTUAL_ENV", "${pwd()}/venv")
    govuk.setEnvar("PATH", "${pwd()}/venv/bin:${env.PATH}")
    // A more recent version of pip is required to install tensorflow
    sh("pip3 install --upgrade pip")
    sh("pip3 install wheel")
  }

  stage('pip wheel') {
    // This speeds up the pip install step by generating wheels which
    // will be cached.
    govuk.setEnvar("WHEELHOUSE", "${env.HOME}/.cache/pip/wheelhouse")
    govuk.setEnvar("PIP_FIND_LINKS", "file://${env.WHEELHOUSE}")
    govuk.setEnvar("PIP_WHEEL_DIR", "${env.WHEELHOUSE}")
    sh 'pip3 wheel --global-option bdist_ext -r python/requirements.txt'
  }

  stage('make pip_install') {
    sh 'make pip_install'
  }

  stage('make check') {
    sh 'make check'
  }

  if (env.BRANCH_NAME == 'master') {
    stage("Push release tag") {
      govuk.pushTag(repoName, env.BRANCH_NAME, 'release_' + env.BUILD_NUMBER)
    }

    stage("Push to Gitlab") {
      try {
        govuk.pushToMirror(repoName, env.BRANCH_NAME, 'release_' + env.BUILD_NUMBER)
      } catch (e) {
      }
    }
  }
}
