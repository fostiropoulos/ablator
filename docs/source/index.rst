:html_theme.sidebar_secondary.remove:
.. ablator documentation master file, created by
   sphinx-quickstart on Tue May  2 20:42:43 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. only:: not html


   ===================================
   Welcome to ABLATOR!
   ===================================

   .. sidebar:: Right Sidebar Title



.. only:: html

   .. raw:: html
         <html lang="en">

            <head>
               <meta charset="utf-8">
               <link rel="stylesheet" href="./_static/css/index.css">
               <link rel="preconnect" href="https://fonts.googleapis.com">
               <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
               <link
                  href="https://fonts.googleapis.com/css2?family=Dancing+Script:wght@400;500;600;700&family=Fira+Code&family=Roboto:wght@300;400;500;700&display=swap"
                  rel="stylesheet">
               <link
                  href="https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;500;600;700&family=Fira+Code:wght@300;400;500;600;700&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap"
                  rel="stylesheet">
               <script>
                  function copyToClipboard(text) {
                        const tempInput = document.createElement('textarea');
                        tempInput.value = text;
                        document.body.appendChild(tempInput);
                        tempInput.select();
                        document.execCommand('copy');
                        document.body.removeChild(tempInput);
                  }
                  function handleClickCopyInstallCode() {
                        copyToClipboard("pip install ablator");
                        const copyIcon = document.getElementById('banner-copy-icon');
                        copyIcon.src = "./_static/img/green_check.png";

                        setTimeout(() => {
                           copyIcon.src = "./_static/img/copy-icon.png";
                        }, 2000);
                  }
               </script>
               <script>
                  function navToPath(path, isNewTab = false) {
                        const href = window.location.href.split('#')[0];
                        const paths = href.split('/');

                        if (paths[paths.length - 1].includes('index')) {
                           paths.pop();
                        }

                        const newHref = paths.join("/") + (path[0] === '/' ? path : '/' + path);
                        if (isNewTab) {
                           window.open(newHref, '_blank');
                        } else {
                           window.location.href = newHref;
                        }
                  }
               </script>
            </head>

            <body>
               <div class="page-root">
                  <div class="page-contents">
                        <div class="banner">
                           <div class="banner-branding">
                              <img class="banner-image" src="./_static/logo.png" alt="ablator-logo" onclick="window.open(`https://ablator.org`)">
                           </div>
                           <div class="banner-texts">
                              <h2>Welcome to ABLATOR Documentation!</h2>
                              <p>ABLATOR is a machine learning library that helps you test and compare several model variants to
                                    find the one that
                                    performs the best; an ablation experiment. Running several experiments in parallel in
                                    cumbersome. ABLATOR will take care
                                    of the hard work.</p>
                              <div class="banner-btn-group">
                                    <button class="custom-btn banner-btn" onclick="window.location.href = `#getting-started`;">
                                       Getting Started
                                    </button>

                                    <div class="banner-codes" onclick="handleClickCopyInstallCode()">
                                       $ pip install ablator

                                       <div class="banner-codes-icon" onclick="handleClickCopyInstallCode()">
                                          <img id="banner-copy-icon" height="100%" width="100%" src="./_static/img/copy-icon.png"
                                                alt="copy">
                                       </div>
                                    </div>
                                    <div class="banner-badges-stack">
                                       <a href="https://github.com/fostiropoulos/ablator" target="_blank">
                                          <img class="banner-icon" src="./_static/img/github-mark.png" alt="github">
                                       </a>
                                       <a href="https://join.slack.com/t/ablator/shared_invite/zt-23ak9ispz-HObgZSEZhyNcTTSGM_EERw"
                                          target="_blank">
                                          <img class="banner-icon" src="./_static/img/slack.png" alt="github">
                                       </a>
                                       <a href="https://discord.gg/9dqThvGnUW" target="_blank">
                                          <img class="banner-icon" src="./_static/img/discord.svg" alt="github">
                                       </a>
                                       <a href="https://twitter.com/ABLATOR_ORG" target="_blank">
                                          <img class="banner-icon" src="./_static/img/twitter.png" alt="github">
                                       </a>
                                    </div>
                              </div>
                           </div>
                        </div>
                        <div class="contents">
                           <div class="contents-texts">
                              <h3>
                                    Quick Overview
                              </h3>
                              <p>
                                    Here is a quick overview of ABLATOR documentations' contents. Usages of ABLATOR are arranged as
                                    following
                                    sections. Please refer to each section for detailed instructions.
                              </p>
                           </div>

                           <div class="contents-grid">

                              <div class="contents-card" onclick="navToPath(`tutorials.html`)">
                                    <div class="card-title">
                                       <h5>
                                          ABLATOR Tutorials
                                       </h5>
                                    </div>

                                    <p>
                                       This section will introduce the comprehensive usages of ABLATOR, including the basic usages and advanced usages. Please refer to this section for detailed instructions.
                                    </p>
                              </div>
                              <div class="contents-card" onclick="navToPath(`modules.html`);">
                                    <div class="card-title">
                                       <h5>
                                          ABLATOR Modules
                                       </h5>
                                    </div>

                                    <p>
                                       ABLATOR is composed of several core modules. This is the section introducing how ABLATOR works with these modules.
                                    </p>
                              </div>
                              <div class="contents-card" onclick="navToPath(`api.reference.html`);">
                                    <div class="card-title">
                                       <h5>
                                          API Reference
                                       </h5>
                                    </div>

                                    <p>
                                       This section is the API reference of ABLATOR. Please refer to this section for detailed usages of ABLATOR modules and functions.
                                    </p>
                              </div>
                              <div class="contents-card" onclick="window.open(`https://github.com/fostiropoulos/ablator-tutorials`);">
                                    <div class="card-title">
                                       <h5>
                                          More Examples
                                       </h5>
                                    </div>
                                    <p>
                                       ABLATOR is capable of handling various types of deep learning experiments. Please visit this
                                       section for more examples of ABLATOR use cases.
                                    </p>
                              </div>
                           </div>
                        </div>

                        <div class="basics" id="getting-started">

                           <h3>
                              Getting Started
                           </h3>
                           <div class="features-grid">
                              <div class="feature-card" onclick="navToPath(`/notebooks/Environment-settings.html`)">
                                    <div class="card-title">
                                       <h5>
                                          Installations
                                       </h5>
                                    </div>
                                    <div class="feature-codes">
                                       $ pip install ablator
                                    </div>

                                    <div class="card-texts">
                                       <p>
                                          Other installation options are also available.
                                       </p>

                                    </div>

                              </div>
                              <div class="feature-card" onclick="navToPath(`/notebooks/GettingStarted.html`)">
                                    <div class="card-title feature-card-title">
                                       <h5>
                                          Quick Start
                                       </h5>
                                    </div>
                                    <div class="card-texts">
                                       <p>
                                          To get started with ABLATOR quickly, try it out in the demo codes below, where a simple
                                          CNN will be
                                          trained and evaluated with ABLATOR.
                                       </p>
                                    </div>
                              </div>
                              <div class="feature-card" onclick="navToPath(`/tutorials.html`)">
                                    <div class="card-title feature-card-title">
                                       <h5>
                                          Learn Basics
                                       </h5>
                                    </div>
                                    <div class="card-texts">
                                       <p>
                                          For more basic usages of ABLATOR, please refer to the Basic Tutorials section below.
                                       </p>
                                    </div>
                              </div>
                           </div>
                        </div>

                        <div class="packages">
                           <div class="contents-texts">
                              <h3>
                                    How ABLATOR Works
                              </h3>
                              <p>
                                    ABLATOR is composed of several core modules. Please refer to this section for
                                    detailed usages of each module of ABLATOR and learn how ABLATOR works.
                              </p>
                           </div>



                           <div class="contents-grid">
                              <div class="feature-card package-card" onclick="navToPath(`/config.html`)">
                                    <div class="card-title">
                                       <h5>
                                          Configuration module
                                       </h5>
                                    </div>

                                    <div class="card-texts">
                                       <p>
                                          In ABLATOR, the configuration system is used as a framework or structure for defining experiments. With this system, ABLATOR creates and sets up experiments, incorporating the appropriate configurations.
                                       </p>
                                    </div>
                              </div>
                              <div class="feature-card package-card" onclick="navToPath(`/training.html`)">
                                    <div class="card-title">
                                       <h5>
                                          Training module
                                       </h5>
                                    </div>

                                    <div class="card-texts">
                                       <p>
                                          Other building blocks of ABLATOR are the training module, which launch the experiment that has been configured with the configuration module.
                                       </p>
                                    </div>
                              </div>
                              <div class="feature-card package-card" onclick="navToPath(`/analysis.html`)">
                                 <div class="card-title">
                                    <h5>
                                       Analysis module
                                    </h5>
                                 </div>

                                 <div class="card-texts">
                                    <p>
                                       The analysis module has tools that allow you to observe the correlation between the
                                       studied hyperparameters and the model's performance.
                                    </p>
                                 </div>
                              </div>
                              <div class="feature-card package-card" onclick="navToPath(`/api.reference.html`)">
                                 <div class="card-title">
                                    <h5>
                                       API Reference
                                    </h5>
                                 </div>

                                 <div class="card-texts">
                                    <p>
                                       For more detailed information about ABLATOR modules and APIs, please refer to the API
                                       Reference.
                                    </p>
                                 </div>
                              </div>
                           </div>
                        </div>
                        <div class="community">
                           <div class="contents-texts">
                              <h3>
                                    ABLATOR Community
                              </h3>

                           </div>

                           <div class="contents-grid">
                              <div class="contents-card community-card"
                                    onclick="window.open('https://github.com/fostiropoulos/ablator')">
                                    <div class="card-title">
                                       <div style="display: flex; align-items: center; gap: 1rem">
                                          <img src="./_static/img/github-mark.png" alt="github" style="height: 40px; width: 40px;">
                                          <h5>
                                                Visit ABLATOR on Github
                                          </h5>
                                       </div>
                                    </div>
                                    <div class="card-texts">
                                       <p>
                                          ABLATOR is an open-source project. Visit ABLATOR on Github to learn more and feel free
                                          to
                                          make your contributions.
                                       </p>
                                    </div>

                              </div>
                              <div class="contents-card community-card" onclick="window.open('https://deep.usc.edu')">
                                    <div class="card-title">
                                       <div style="display: flex; align-items: center; gap: 1rem">
                                          <img src="./_static/img/group_logo.png" alt="github" style="height: 40px; width: 45px;">
                                          <h5>
                                                Meet the developers
                                          </h5>
                                       </div>
                                    </div>
                                    <div class="card-texts">
                                       <p>
                                          ABLATOR is developed and maintained by Deep USC Research Group from University of
                                          Southern California.
                                       </p>
                                    </div>
                              </div>
                           </div>
                           <h5>
                              Follow ABLATOR on social media
                           </h5>
                           <div class="features-grid social-grid">
                              <div class="contents-card social-card"
                                    onclick="window.open('https://join.slack.com/t/ablator/shared_invite/zt-23ak9ispz-HObgZSEZhyNcTTSGM_EERw', '_blank')">
                                    <div class="card-title">
                                       <img src="./_static/img/slack.png" alt="slack">
                                    </div>
                                    <div class="card-texts">
                                       <p>
                                          Slack Workspace
                                       </p>
                                    </div>
                              </div>
                              <div class="contents-card social-card"
                                    onclick="window.open('https://discord.gg/9dqThvGnUW', '_blank')">
                                    <div class="card-title">
                                       <img src="./_static/img/discord.svg" alt="slack">
                                    </div>
                                    <div class="card-texts">
                                       <p>
                                          Discord Community
                                       </p>
                                    </div>
                              </div>
                              <div class="contents-card social-card"
                                    onclick="window.open('https://twitter.com/ABLATOR_ORG', '_blank')">
                                    <div class="card-title">
                                       <img src="./_static/img/twitter.png" alt="slack">
                                    </div>
                                    <div class="card-texts">
                                       <p>
                                          ABLATOR Official Twitter
                                       </p>
                                    </div>
                              </div>
                           </div>
                        </div>
                  </div>
               </div>
            </body>

         </html>


.. only:: sidebar

   .. toctree::
      :maxdepth: 1

      Quick Start <notebooks/GettingStarted.ipynb>
      Tutorials <tutorials>
      Modules <modules>
      API Reference <api.reference>
      More Examples <https://github.com/fostiropoulos/ablator-tutorials>

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
