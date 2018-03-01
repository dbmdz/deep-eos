# Contribution

Thank you for your interest in contributing to the *deep-eos* project.
This guide details how to contribute in a way that is efficient for everyone.

# Issue tracker

Whenever you found a problem or want to file a bug report please
use the *deep-eos* issue tracker [here](https://github.com/stefan-it/deep-eos/issues).

## Issue tracker guidelines

Please search the issue tracker first before submitting your own. A bug report
should be described as detailed as possible, e.g. if you write a bug report
please provide detailed steps to reproduce the bug or provide relevant logs and
error messages.

All issues should be written in English.

# Merge requests

We welcome merge requests with fixes and improvements to the *deep-eos*
code, tests, and/or further documentation.

## Merge request guidelines

If you can, please submit a merge request with the fix or improvements including
unittests. In general bug fixes that include a regression test should be merged
quickly, while new shiny features without proper unittests are very likely to
to be merged later. The workflow to make a merge request is:

* Fork the *deep-eos* repository from GitHub: `https://github.com/stefan-it/deep-eos.git`

* Create a new feature branch, e.g. labeled with `feature/meaningful-description`.
  If there's any issue available, please include the issue number in the branch
  name like: `feature/issue-42`

* Write code/bug fixes new features and please also include test cases for them

* Commit often and push to your feature branch

* Provide a merge request on GitHub

* In this merge request you should describe all implemented features and please
  link any relevant issues.

* Be prepared to answer questions and incorporate feedback after your submission

Please keep the change in a single merge request as small as possible.

# Style guides

Please refer to our [coding style](CODINGSTYLE.md) document when writing code in
*Python*.

# Sources

This guide is heavily inspired by the excellent [GitLab](https://gitlab.com/groups/gitlab-org)
constributing guide, which is available [here](https://gitlab.com/gitlab-org/gitlab-ce/blob/master/CONTRIBUTING.md).
