#!/usr/bin/env bash
# Reproducible install of the community neuroimaging reference tools (AFNI +
# FSL) used as registration gold standards by the perf-bench registration
# cases. These live on /scratch, which is EPHEMERAL (it can vanish if the box
# is reprovisioned) -- this script (committed to the repo) is the durable
# recipe to recreate them. Idempotent-ish: skips a tool whose marker binary is
# already present.
#
#   AFNI       -> $DEST/abin        (3dvolreg: the volreg / realign standard)
#   FSL        -> $DEST/fsl         (mcflirt: volreg; flirt -cost bbr: BBR)
#   FreeSurfer -> $DEST/freesurfer  (bbregister: a 2nd BBR ref) -- OPT-IN, huge
#
# After running, point the perf-bench providers at the tools (the wrappers read
# these, NOT $PATH -- robust to a missing shell profile):
#   export NPERF_AFNI_DIR=$DEST/abin
#   export NPERF_FSL_DIR=$DEST/fsl
#   export NPERF_FREESURFER_DIR=$DEST/freesurfer   # if FREESURFER=1
# (the script appends these to $DEST/neuro_refs_env.sh for sourcing).
#
# Usage:  bash tools/setup_neuro_refs.sh [DEST]        # DEST default /scratch/nperf
#         AFNI_ONLY=1 bash tools/setup_neuro_refs.sh   # or FSL_ONLY=1
#         FREESURFER=1 bash tools/setup_neuro_refs.sh  # ADD FreeSurfer (~9.5 GB
#                                                      # tarball, slow MGH mirror,
#                                                      # many hours); or
#         FREESURFER_ONLY=1 ...                        # FreeSurfer alone
#
# FreeSurfer notes: precompiled centos7 build (glibc 2.17 -> runs on this glibc
# 2.39 box). bbregister is a tcsh script with a hardcoded `#!/bin/tcsh` shebang,
# so we install tcsh on /scratch (conda-forge via micromamba) and symlink it to
# /bin/tcsh. A FREE license.txt (https://surfer.nmr.mgh.harvard.edu/registration.html)
# must be dropped at $FREESURFER_HOME/license.txt before bbregister will run.
#
# Validated 2026-06-11 on Ubuntu 24.04 / glibc 2.39 / x86_64 with:
#   AFNI  26.1.04   (linux_ubuntu_24_64 -- the tarball is rolling "latest"; the
#                    version is recorded for provenance, not pinned in the URL)
#   FSL   6.0.7.22  (pinned via the installer's -V flag below). mcflirt lands
#                    at $FSLDIR/bin; measured I/O floor (fslmaths) ~67% of the
#                    mcflirt wall-clock at T=50/48^3 -> the economic report
#                    subtracts it (compute = tool - iofloor).
set -euo pipefail

DEST="${1:-/scratch/nperf}"
AFNI_URL="https://afni.nimh.nih.gov/pub/dist/tgz/linux_ubuntu_24_64.tgz"
FSL_INSTALLER="https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py"
FSL_VERSION="${FSL_VERSION:-6.0.7.22}"   # pin for reproducibility (see header)
FS_VERSION="${FS_VERSION:-7.4.1}"        # FreeSurfer pin (see header)
# centos7 build: glibc 2.17 -> portable onto this glibc 2.39 box.
FS_URL="https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/${FS_VERSION}/freesurfer-linux-centos7_x86_64-${FS_VERSION}.tar.gz"
MICROMAMBA_URL="https://micro.mamba.pm/api/micromamba/linux-64/latest"
PYTHON="${NPERF_PYTHON:-/scratch/nperf/venv/bin/python}"

mkdir -p "$DEST" "$DEST/tmp"
export TMPDIR="$DEST/tmp"   # keep temp off the tiny root overlay

# -- AFNI: precompiled binaries (no tcsh needed to RUN 3dvolreg) --------------
if [ -z "${FSL_ONLY:-}" ] && [ -z "${FREESURFER_ONLY:-}" ]; then
  if [ -x "$DEST/abin/3dvolreg" ]; then
    echo "AFNI: 3dvolreg already present at $DEST/abin -- skipping."
  else
    echo "AFNI: downloading + extracting -> $DEST/abin"
    curl -fL --retry 3 -o "$DEST/afni_u24.tgz" "$AFNI_URL"
    rm -rf "$DEST/abin" && mkdir -p "$DEST/abin"
    # --no-same-owner: we are not the tarball's original uid; without it tar
    # aborts mid-extract on the first chown failure (incomplete install).
    tar --no-same-owner -xzf "$DEST/afni_u24.tgz" -C "$DEST/abin" \
        --strip-components=1
    rm -f "$DEST/afni_u24.tgz"
  fi
  "$DEST/abin/3dvolreg" -help >/dev/null 2>&1 \
    && echo "AFNI: 3dvolreg OK ($("$DEST/abin/afni" -ver 2>/dev/null | head -1))"
fi

# -- FSL: conda-based, self-contained (mcflirt + flirt -bbr + epi_reg) --------
if [ -z "${AFNI_ONLY:-}" ] && [ -z "${FREESURFER_ONLY:-}" ]; then
  if [ -x "$DEST/fsl/bin/mcflirt" ]; then
    echo "FSL: mcflirt already present at $DEST/fsl -- skipping."
  else
    echo "FSL: installing $FSL_VERSION -> $DEST/fsl (no shell mod, no CUDA)"
    curl -fL --retry 3 -o "$DEST/fslinstaller.py" "$FSL_INSTALLER"
    # -d dest, -n no-env, -r skip-registration, -c none skip-CUDA-libs,
    # -V pin the version. Non-interactive.
    "$PYTHON" "$DEST/fslinstaller.py" -d "$DEST/fsl" -n -r -c none \
        -V "$FSL_VERSION"
    rm -f "$DEST/fslinstaller.py"
  fi
  "$DEST/fsl/bin/mcflirt" 2>&1 | grep -qi mcflirt \
    && echo "FSL: mcflirt OK ($(cat "$DEST/fsl/etc/fslversion" 2>/dev/null))"
fi

# -- FreeSurfer: bbregister (a 2nd BBR ref). OPT-IN (FREESURFER=1) -- ~9.5 GB --
if [ -n "${FREESURFER:-}${FREESURFER_ONLY:-}" ]; then
  if [ -x "$DEST/freesurfer/bin/bbregister" ]; then
    echo "FreeSurfer: bbregister already present at $DEST/freesurfer -- skipping."
  else
    echo "FreeSurfer: downloading $FS_VERSION (~9.5 GB, slow MGH mirror) -> $DEST/freesurfer"
    # -C - resumes a partial download (vital over a multi-hour transfer).
    curl -fL -C - --retry 8 --retry-delay 15 --retry-connrefused \
        -o "$DEST/freesurfer-$FS_VERSION.tar.gz" "$FS_URL"
    # The tarball's top dir is already `freesurfer/` (entries `./freesurfer/...`)
    # -> extract straight into $DEST, NO --strip-components (strip=1 only eats
    # the leading `.` and double-nests freesurfer/freesurfer). --no-same-owner:
    # we are not the tarball's uid (as for AFNI above).
    rm -rf "$DEST/freesurfer"
    tar --no-same-owner -xzf "$DEST/freesurfer-$FS_VERSION.tar.gz" -C "$DEST"
    rm -f "$DEST/freesurfer-$FS_VERSION.tar.gz"
  fi
  # bbregister is `#!/bin/tcsh`: provide tcsh on /scratch (conda-forge) + a
  # /bin/tcsh symlink (0 bytes on root; /scratch is ephemeral, so this recipe
  # recreates both). Skip if a working /bin/tcsh already exists.
  if ! /bin/tcsh -c 'exit 0' 2>/dev/null; then
    [ -x "$DEST/bin/micromamba" ] || { mkdir -p "$DEST/bin"; \
      curl -Ls --retry 3 "$MICROMAMBA_URL" | tar -xj -C "$DEST" bin/micromamba; }
    [ -x "$DEST/tcsh-env/bin/tcsh" ] || MAMBA_ROOT_PREFIX="$DEST/mamba" \
      "$DEST/bin/micromamba" create -y -p "$DEST/tcsh-env" -c conda-forge tcsh
    ln -sf "$DEST/tcsh-env/bin/tcsh" /bin/tcsh 2>/dev/null \
      && echo "FreeSurfer: tcsh -> /bin/tcsh ($("$DEST/tcsh-env/bin/tcsh" --version 2>&1 | cut -d' ' -f1-2))" \
      || echo "FreeSurfer: WARN /bin not writable -- bbregister needs tcsh at /bin/tcsh (root)."
  fi
  # FreeSurfer's centos7 binaries need runtime libs this box lacks (libgomp for
  # OpenMP; an X11/GL stack for tkregister2). Resolve them SURGICALLY into
  # $FS/lib.extra (symlinks) -- do NOT put a whole conda lib dir on
  # LD_LIBRARY_PATH (it would shadow FreeSurfer's own libstdc++/libz and break
  # the binaries that already work). Sources: the conda tcsh-env (libgomp,
  # libXmu) + the FSL env's GL/X11 stack. The wrappers add $FS/lib.extra to
  # LD_LIBRARY_PATH (see neuro_refs_env.sh).
  FS="$DEST/freesurfer"; mkdir -p "$FS/lib.extra"
  MAMBA_ROOT_PREFIX="$DEST/mamba" "$DEST/bin/micromamba" install -y \
      -p "$DEST/tcsh-env" -c conda-forge libgomp xorg-libxmu >/dev/null 2>&1 || true
  for _round in 1 2 3 4 5 6; do
    _missing=$(for _b in mri_coreg mri_segreg mri_convert tkregister2; do
                 LD_LIBRARY_PATH="$FS/lib.extra" ldd "$FS/bin/$_b" 2>&1; done \
               | awk '/not found/{print $1}' | sort -u)
    [ -z "$_missing" ] && break
    for _lib in $_missing; do
      for _d in "$DEST/tcsh-env/lib" "$DEST/fsl/lib"; do
        _f=$(find "$_d" -maxdepth 1 -name "$_lib" 2>/dev/null | head -1)
        [ -n "$_f" ] && { ln -sf "$_f" "$FS/lib.extra/$_lib"; break; }
      done
    done
  done
  LD_LIBRARY_PATH="$FS/lib.extra" "$FS/bin/mri_coreg" --version >/dev/null 2>&1 \
    && echo "FreeSurfer: core registration binaries OK (mri_coreg, mri_segreg)" \
    || echo "FreeSurfer: WARN core binaries still missing libs -- check $FS/lib.extra"
  if [ -f "$DEST/freesurfer/license.txt" ]; then
    echo "FreeSurfer: installed + licensed ($(cat "$DEST/freesurfer/build-stamp.txt" 2>/dev/null))"
  else
    echo "FreeSurfer: installed ($(cat "$DEST/freesurfer/build-stamp.txt" 2>/dev/null)) -- NO LICENSE yet."
    echo "  Drop a free license.txt at $DEST/freesurfer/license.txt before bbregister"
    echo "  will run: https://surfer.nmr.mgh.harvard.edu/registration.html"
  fi
fi

# -- nibabel for the NIfTI round-trip wrappers (write array -> run -> read) ---
"$PYTHON" -c "import nibabel" 2>/dev/null \
  || "$PYTHON" -m pip install --quiet nibabel

# -- emit the env the perf-bench providers read ------------------------------
cat > "$DEST/neuro_refs_env.sh" <<EOF
# sourced by perf-bench runs to locate the community neuro refs (generated by
# tools/setup_neuro_refs.sh). The wrappers use these absolute dirs, not \$PATH.
export NPERF_AFNI_DIR="$DEST/abin"
export NPERF_FSL_DIR="$DEST/fsl"
export FSLDIR="$DEST/fsl"
export FSLOUTPUTTYPE=NIFTI_GZ
export PATH="$DEST/abin:$DEST/fsl/bin:\$PATH"
EOF
# FreeSurfer is opt-in; only emit its env if it actually got installed.
if [ -x "$DEST/freesurfer/bin/bbregister" ]; then
  cat >> "$DEST/neuro_refs_env.sh" <<EOF
export NPERF_FREESURFER_DIR="$DEST/freesurfer"
export FREESURFER_HOME="$DEST/freesurfer"
# bbregister + recon-all need the fuller env (SUBJECTS_DIR, PERL5LIB, PATH ...):
[ -f "\$FREESURFER_HOME/SetUpFreeSurfer.sh" ] && \\
  source "\$FREESURFER_HOME/SetUpFreeSurfer.sh" >/dev/null 2>&1 || true
# the surgical runtime libs this box lacks (libgomp + X11/GL for tkregister2):
export LD_LIBRARY_PATH="\$FREESURFER_HOME/lib.extra:\${LD_LIBRARY_PATH:-}"
EOF
fi
echo "wrote $DEST/neuro_refs_env.sh  (source it, or set NPERF_AFNI_DIR / NPERF_FSL_DIR)"
echo "DONE."
