"""
===============================================================================
 IRI-2020 Space-Weather Pipeline — Daily foF2 with User-Supplied F10.7
===============================================================================
 Runs IRI-2020 for each (DayOfYear, Hour, Station) record using the ACTUAL
 daily F10.7 from the input CSV (not IRI's internal index files).

 IRI JF switches used:
   JF(5)  toggled  → CCIR (.TRUE.) or URSI (.FALSE.) for foF2
   JF(25) = .FALSE.→ daily F10.7 from user input via OARR(41)
   JF(32) = .FALSE.→ 81-day avg F10.7 from user input via OARR(46)
   MMDD   = -DOY   → day-of-year convention (not month+day)
   IYYYY  = 2019

 Output: original CSV with two appended columns — foF2_C and foF2_U
===============================================================================
"""

import subprocess, os, sys, time, re
import numpy as np
import pandas as pd
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

WORK_DIR   = Path("/home/ayyaz/iri00")
INPUT_CSV  = WORK_DIR / "FINAL_MASTER.csv"
OUTPUT_CSV = WORK_DIR / "FINAL_MASTER_IRI.csv"
YEAR       = 2019
CHUNK_SIZE = 40000   # small dataset — one chunk is fine

# ═══════════════════════════════════════════════════════════════════════════════
# Fortran batch driver — accepts daily F10.7 as user input
# ═══════════════════════════════════════════════════════════════════════════════

FORTRAN_DRIVER = r"""
program iri_sw_batch
!---------------------------------------------------------------
! Space-weather batch driver for IRI-2020
!
! Reads: lat lon year doy hour f107 f107_81
! Uses user-supplied F10.7 daily and 81-day avg (JF(25)=F, JF(32)=F)
! MMDD = -DOY (day-of-year convention)
! Runs IRI twice (CCIR / URSI) and outputs foF2 for each
!---------------------------------------------------------------
implicit none
logical :: jf(50)
integer, parameter :: jmag = 0
integer :: iyyyy, mmdd, ios, nrec
real :: glat, glon, glat0, glon0, dhour
real :: oarr(100), outf(20,1000)
real :: foF2_CCIR, foF2_URSI
real :: f107_in, f107_81_in
integer :: yr, doy, hr

call read_ig_rz
call readapf107
nrec = 0

do
   read(*,*, iostat=ios) glat, glon, yr, doy, hr, f107_in, f107_81_in
   if (ios /= 0) exit
   glat0 = glat; glon0 = glon
   iyyyy = yr
   mmdd  = -doy                     ! negative = day-of-year
   dhour = real(hr) + 25.0          ! +25 = UT convention

   ! --- CCIR run ---
   call setup_jf(jf)
   jf(5) = .true.                   ! CCIR coefficients
   oarr = -1.0
   oarr(41) = f107_in               ! user daily F10.7
   oarr(46) = f107_81_in            ! user 81-day avg F10.7
   call IRI_SUB(JF,JMAG,glat,glon,IYYYY,MMDD,DHOUR, &
                300.,300.,1.,OUTF,OARR)
   foF2_CCIR = oarr(100)

   ! --- URSI run ---
   glat = glat0; glon = glon0
   call setup_jf(jf)
   jf(5) = .false.                  ! URSI coefficients
   oarr = -1.0
   oarr(41) = f107_in
   oarr(46) = f107_81_in
   call IRI_SUB(JF,JMAG,glat,glon,IYYYY,MMDD,DHOUR, &
                300.,300.,1.,OUTF,OARR)
   foF2_URSI = oarr(100)

   write(*,'(F8.3,1X,F8.3)') foF2_CCIR, foF2_URSI
   nrec = nrec + 1
   if (mod(nrec,10000)==0) write(0,'(A,I0)') '  Processed: ', nrec
enddo
write(0,'(A,I0)') 'Total: ', nrec

contains

subroutine setup_jf(jf)
   logical, intent(out) :: jf(50)
   jf = .true.
   jf(2)  = .false.   ! skip Te, Ti
   jf(3)  = .false.   ! skip Ni
   jf(4)  = .false.   ! B0: ABT-2009
   jf(6)  = .false.   ! ion comp: RBV-2010
   jf(12) = .false.   ! no topside
   jf(22) = .false.   ! units: m^-3
   jf(23) = .false.   ! default
   jf(25) = .false.   ! *** F10.7 daily from OARR(41) ***
   jf(28) = .false.   ! no spread-F
   jf(29) = .false.   ! no drift
   jf(30) = .false.   ! default
   jf(32) = .false.   ! *** F10.7_81 from OARR(46) ***
   jf(33) = .false.   ! default
   jf(34) = .false.   ! suppress messages
   jf(35) = .false.   ! default
   jf(39) = .false.   ! hmF2: SHU-2015
   jf(40) = .false.   ! default
   jf(47) = .false.   ! default
end subroutine

end program
"""


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 : Patch irisub.for and compile
# ═══════════════════════════════════════════════════════════════════════════════

def patch_irisub():
    irisub = WORK_DIR / "irisub.for"
    text = irisub.read_text(encoding="latin-1")
    for line in text.splitlines():
        if line.strip().lower().startswith("oarr(100)") and "fof2" in line.lower():
            print("  irisub.for already patched")
            return True
    match = re.search(r'^(\s*3330\s+CONTINUE)', text, re.MULTILINE)
    if match:
        text = text.replace(match.group(1),
                            "      oarr(100) = foF2\n" + match.group(1), 1)
        irisub.write_text(text, encoding="latin-1")
        print("  Patched irisub.for: inserted oarr(100) = foF2")
        return True
    print("  ERROR: cannot find 3330 CONTINUE in irisub.for")
    return False


def compile_iri():
    print("=" * 60)
    print("STEP 1: Compile IRI-2020 (space-weather driver)")
    print("=" * 60)

    essential = ["irisub.for", "irifun.for", "iritec.for", "iridreg.for",
                 "igrf.for", "cira.for", "iriflip.for", "rocdrift.for",
                 "ccir11.asc", "ursi11.asc", "ig_rz.dat", "apf107.dat"]
    missing = [f for f in essential if not (WORK_DIR / f).exists()]
    if missing:
        print(f"  Missing: {missing}")
        sys.exit(1)
    print(f"  All IRI files present")

    if not patch_irisub():
        sys.exit(1)

    driver = WORK_DIR / "iri_sw_batch.f90"
    driver.write_text(FORTRAN_DRIVER)
    print(f"  Wrote driver: {driver.name}")

    exe = WORK_DIR / "iri_sw_batch"
    src = ["iri_sw_batch.f90", "irisub.for", "irifun.for", "iritec.for",
           "iridreg.for", "igrf.for", "cira.for", "iriflip.for", "rocdrift.for"]
    cmd = ["gfortran", "-O3", "-w", "-fallow-argument-mismatch",
           "-std=legacy", "-o", str(exe)] + src

    print("  Compiling ...")
    r = subprocess.run(cmd, cwd=WORK_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        print("  FAILED:\n" + r.stderr)
        sys.exit(1)
    print(f"  Compiled: {exe.name}")

    # Smoke test with user F10.7
    print("  Smoke test (Jicamarca, DOY 180, 12 UT, F10.7=70) ...")
    test_in = "-12.0 283.2 2019 180 12 70.0 69.5\n"
    t = subprocess.run([str(exe)], input=test_in, capture_output=True,
                       text=True, cwd=str(WORK_DIR), timeout=30)
    if t.returncode != 0 or not t.stdout.strip():
        print(f"  FAILED: {t.stderr}")
        sys.exit(1)
    parts = t.stdout.strip().split()
    ccir, ursi = float(parts[0]), float(parts[1])
    if ccir <= 0 or ursi <= 0:
        print(f"  FAILED: foF2_CCIR={ccir}, foF2_URSI={ursi}")
        sys.exit(1)
    print(f"  PASSED: foF2_CCIR={ccir:.3f}  foF2_URSI={ursi:.3f} MHz")
    return exe


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 : Load CSV and compute F10.7_81
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_prepare():
    print("\n" + "=" * 60)
    print("STEP 2: Load data and compute F10.7_81")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV)
    print(f"  Loaded: {len(df):,} rows, {df['Station'].nunique()} stations")
    print(f"  DOY range: {df['DayOfYear'].min()}–{df['DayOfYear'].max()}")

    # F10.7 is one value per DOY (same across stations/hours)
    f107_daily = df.groupby('DayOfYear')['F10.7'].first().sort_index()

    # 81-day centered running average (min_periods=41 for edge days)
    f107_81 = f107_daily.rolling(81, center=True, min_periods=41).mean()

    # Map back to full dataframe
    df['F107_81'] = df['DayOfYear'].map(f107_81)
    print(f"  F10.7 daily range:  {f107_daily.min():.1f} – {f107_daily.max():.1f}")
    print(f"  F10.7_81 range:     {df['F107_81'].min():.1f} – {df['F107_81'].max():.1f}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 : Run IRI
# ═══════════════════════════════════════════════════════════════════════════════

def run_iri(exe, df):
    print("\n" + "=" * 60)
    print("STEP 3: Running IRI-2020 (user F10.7, DOY-based)")
    print("=" * 60)

    total = len(df)
    print(f"  Records: {total:,}")
    print(f"  Estimated time: ~{total * 5 / 1000:.0f}s")

    # Build input: lat lon year doy hour f107 f107_81
    input_lines = []
    for _, r in df.iterrows():
        input_lines.append(
            f"{r['Latitude']:.1f} {r['Longitude']:.1f} "
            f"{YEAR} {int(r['DayOfYear'])} {int(r['Hour'])} "
            f"{r['F10.7']:.1f} {r['F107_81']:.1f}"
        )

    all_output = []
    t0 = time.time()

    for start in range(0, total, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total)
        chunk = "\n".join(input_lines[start:end]) + "\n"
        try:
            result = subprocess.run(
                [str(exe)], input=chunk, capture_output=True,
                text=True, cwd=str(WORK_DIR), timeout=600)
            if result.stdout.strip():
                all_output.extend(result.stdout.strip().split("\n"))
        except Exception as e:
            print(f"  WARNING: chunk {start}-{end} failed: {e}")

        done = len(all_output)
        dt = time.time() - t0
        print(f"  {done:,} / {total:,} done  ({dt:.0f}s)", flush=True)

    dt = time.time() - t0
    print(f"\n  Completed: {len(all_output):,} records in {dt:.0f}s")

    # Parse
    foF2_C = []
    foF2_U = []
    for line in all_output:
        parts = line.split()
        if len(parts) >= 2:
            foF2_C.append(float(parts[0]))
            foF2_U.append(float(parts[1]))
        else:
            foF2_C.append(np.nan)
            foF2_U.append(np.nan)

    # Pad if some records failed
    while len(foF2_C) < total:
        foF2_C.append(np.nan)
        foF2_U.append(np.nan)

    bad = sum(1 for v in foF2_C if v is not None and v <= 0)
    print(f"  Validation: {bad} records with foF2 <= 0")

    return foF2_C[:total], foF2_U[:total]


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 : Merge and write CSV
# ═══════════════════════════════════════════════════════════════════════════════

def merge_and_write(df, foF2_C, foF2_U):
    print("\n" + "=" * 60)
    print("STEP 4: Merge IRI results and write CSV")
    print("=" * 60)

    df['foF2_C'] = foF2_C
    df['foF2_U'] = foF2_U

    # Drop the helper column
    df.drop(columns=['F107_81'], inplace=True)

    # Write — keep original column order + new columns at end
    df.to_csv(OUTPUT_CSV, index=False, float_format='%.3f')

    print(f"  Written: {OUTPUT_CSV.name}")
    print(f"  Rows:    {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    # Quick stats
    print(f"\n  --- Summary ---")
    print(f"  foF2 observed: {df['foF2'].min():.2f} – {df['foF2'].max():.2f} MHz")
    print(f"  foF2_C (CCIR): {df['foF2_C'].min():.2f} – {df['foF2_C'].max():.2f} MHz")
    print(f"  foF2_U (URSI): {df['foF2_U'].min():.2f} – {df['foF2_U'].max():.2f} MHz")

    # Per-station RMSE
    print(f"\n  --- Station accuracy ---")
    for stn in sorted(df['Station'].unique()):
        sub = df[df['Station'] == stn].dropna(subset=['foF2_C'])
        if len(sub) > 0:
            rmse_c = np.sqrt(((sub['foF2'] - sub['foF2_C'])**2).mean())
            rmse_u = np.sqrt(((sub['foF2'] - sub['foF2_U'])**2).mean())
            corr_c = sub['foF2'].corr(sub['foF2_C'])
            corr_u = sub['foF2'].corr(sub['foF2_U'])
            print(f"  {stn:<14s} CCIR: RMSE={rmse_c:.2f} r={corr_c:.3f}  |  "
                  f"URSI: RMSE={rmse_u:.2f} r={corr_u:.3f}  (n={len(sub):,})")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("+" + "=" * 58 + "+")
    print("|  IRI-2020 Space-Weather foF2 — User F10.7, DOY-based   |")
    print("|  5 American-sector stations, Year 2019                  |")
    print("+" + "=" * 58 + "+\n")

    t0 = time.time()
    exe = compile_iri()
    df = load_and_prepare()
    foF2_C, foF2_U = run_iri(exe, df)
    merge_and_write(df, foF2_C, foF2_U)

    print(f"\n{'='*60}")
    print(f"  DONE in {(time.time()-t0)/60:.1f} minutes")
    print(f"  Output: {OUTPUT_CSV}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()