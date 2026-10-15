# Preparing the public release

1. Confirm which scripts and supplementary panels generated the final submitted
   figures. Record any manual composition, editing, exclusions and cohort sizes.
2. Confirm an access mechanism for the exact processed data, metadata, controls
   and caches. Add persistent identifiers or precise request instructions and
   clarify access restrictions. Data/artwork must remain outside this Git repo
   unless redistribution is separately approved.
3. Choose a code license with all relevant authors/institutional owners. Add
   `LICENSE` and update README licensing language. No license is selected yet.
4. Validate installation in a new environment, run all manuscript figure
   commands, compare to the submitted figures and update `VALIDATION.md` with
   measured runtimes, resources, discrepancies and any known exclusions.
5. The public repository is
   [JudahGazit/two-tiers-mammalian-brain](https://github.com/JudahGazit/two-tiers-mammalian-brain).
   This local directory has independent Git metadata and no connection to the
   private remote. Review `git status` and `git diff --cached --stat` before
   committing. Example commands for subsequent updates, run from this directory:

   ```text
   git add .
   git commit -m "Update manuscript reproduction code"
   git push -u origin main
   ```

6. After validation, tag the manuscript version (for example `v1.0.0`) and make
   a release. Archive that exact version in Zenodo or an equivalent persistent
   repository, then add its DOI and the repository URL to the manuscript's Code
   availability statement and reference list. Add the final article citation
   to this repository when known.

Suggested Code availability wording, to complete with actual identifiers:

> The custom analysis and figure-generation code is available at [repository
> URL]. The version associated with this manuscript is archived at [version DOI].
> Input data and artwork are subject to the access conditions described in the
> Data availability statement and repository documentation.

If a license is selected, state it explicitly. Do not claim unrestricted access
to external data/artwork or a fully verified reproduction until confirmed.
