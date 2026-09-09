import numpy as np 
import healpy as hp 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.stats import rankdata 

import numpy as np 
import healpy as hp 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.stats import rankdata 


def run_spatial_permutation_test_new(
    inputGroupingVar, 
    inputMomZero, 
    inputMomOne, 
    inputMomTwo, 
    badDataMask, 
    coarseNside=16, 
    fineNside=128, 
    numTrials=10000,
    minPixelCount=10, 
    makePlot=True,
    generateStatsTable=False
): 
    """
    Executes spatial Monte Carlo permutation and block bootstrap test on moment maps grouped by values of another moment map. 

    Returns: A dictionary containing the computed stats Dataframe (on request), and the raw null/bootstrap
    arrays for plotting or more analysis.
    """
    print("Generating grids and mapping geometry") 

    groupingClean = np.where(
        (inputGroupingVar==hp.UNSEEN) | badDataMask, np.nan, inputGroupingVar
    )
    M0Clean = np.where((inputMomZero==hp.UNSEEN)|badDataMask, np.nan, inputMomZero)
    M1Clean = np.where((inputMomOne==hp.UNSEEN)|badDataMask, np.nan, inputMomOne)
    M2Clean = np.where((inputMomTwo==hp.UNSEEN)|badDataMask, np.nan, inputMomTwo)

    nPixFine = hp.nside2npix(fineNside)
    allFinePixIDs = np.arange(nPixFine) 
    fineTheta, finePhi = hp.pix2ang(fineNside, allFinePixIDs)
    allPatchIDsFull = hp.ang2pix(coarseNside, fineTheta, finePhi)

    uniquePatches, patchCounts = np.unique(
        allPatchIDsFull[~np.isnan(groupingClean)], return_counts=True
    )
    validPatchesList = uniquePatches[patchCounts >= minPixelCount]

    validPixelMask   = np.isin(allPatchIDsFull, validPatchesList) & ~np.isnan(groupingClean)
    groupingValid    = groupingClean[validPixelMask]
    m0Valid          = M0Clean[validPixelMask]
    m1Valid          = M1Clean[validPixelMask] 
    m2Valid          = M2Clean[validPixelMask] 
    patchValid       = allPatchIDsFull[validPixelMask]

    patchIndicesMap = {
        patchID: np.where(patchValid == patchID)[0] for patchID in validPatchesList
    }

    uniqueP, inverseP, countsP = np.unique(
        patchValid, return_inverse=True, return_counts=True
    )
    pixelsPerPatchArray = countsP[inverseP]

    latBinEdges = np.array([0.0, 30.0, 60.0, 90.0])
    patchTheta, patchPhi = hp.pix2ang(coarseNside, validPatchesList)
    patchLatDeg = 90.0 - np.degrees(patchTheta)
    patchAbsLatDeg = np.abs(patchLatDeg)
    patchBinIndex = np.digitize(patchAbsLatDeg, latBinEdges) - 1
    patchBinIndex = np.clip(patchBinIndex, 0, len(latBinEdges) - 2)

    patchesByBin = {}
    
    for binID in range(len(latBinEdges)-1): 
        patchesByBin[binID] = validPatchesList[patchBinIndex == binID]

    # FIX: Un-indented everything below to pull it out of the binID loop
    patchSortIdx = np.searchsorted(validPatchesList, patchValid)
    pixelBinID = patchBinIndex[patchSortIdx]

    binThresholds = {}
    lowMask = np.zeros(len(groupingValid), dtype=bool)
    highMask = np.zeros(len(groupingValid), dtype=bool)

    for binID in range(len(latBinEdges) - 1):
        inBin = pixelBinID == binID
        if inBin.sum() < minPixelCount:
            continue
        q15Bin, q85Bin = np.nanpercentile(groupingValid[inBin], [15.0, 85.0])
        binThresholds[binID] = (q15Bin, q85Bin)
        lowMask  |= inBin & (groupingValid <= q15Bin)
        highMask |= inBin & (groupingValid >= q85Bin)

    obsDeltaM0 = np.nanmean(m0Valid[highMask]) - np.nanmean(m0Valid[lowMask])
    obsDeltaM1 = np.nanmean(m1Valid[highMask]) - np.nanmean(m1Valid[lowMask]) 
    obsDeltaM2 = np.nanmean(m2Valid[highMask]) - np.nanmean(m2Valid[lowMask])

    print(f"Running permutations with {numTrials} trials")
    nullDeltaM0, nullDeltaM1, nullDeltaM2 = [], [], []

    for trial in range(numTrials):
        patchMapping = {} 
        for binID in range(len(latBinEdges) -1): 
            binPatches = patchesByBin[binID]
            if len(binPatches) == 0: 
                continue 
            shuffledBinPatches = np.random.permutation(binPatches) 
            patchMapping.update(dict(zip(binPatches, shuffledBinPatches)))

        scrambledIndices = np.zeros_like(patchValid, dtype=int)
        for origPatch in validPatchesList: 
            newPatch = patchMapping[origPatch]
            origPixIdx = patchIndicesMap[origPatch]
            newPixIdx = patchIndicesMap[newPatch]

            if len(origPixIdx) <= len(newPixIdx): 
                scrambledIndices[origPixIdx] = np.random.choice(
                    newPixIdx, size=len(origPixIdx), replace=False
                )
            else: 
                scrambledIndices[origPixIdx] = np.random.choice(
                    newPixIdx, size=len(origPixIdx), replace=True
                )
                
        m0Null = m0Valid[scrambledIndices] 
        m1Null = m1Valid[scrambledIndices] 
        m2Null = m2Valid[scrambledIndices]

        # randNoise = np.random.rand(len(patchValid))
        # ranking   = rankdata(randNoise + patchValid * 2.0,  method='ordinal')

        # min_rank_per_patch = (
        #     np.minimum.reduceat(ranking, np.r_[0, np.cumsum(countsP)[:-1]])
        #     if len(countsP) > 0
        #     else 1
        # )
        
        # pixel_rank_in_patch = ranking - min_rank_per_patch[inverseP] + 1
        # pixel_percentile_in_patch = pixel_rank_in_patch / pixelsPerPatchArray

        # randMask1 = (pixel_percentile_in_patch > 0.00) & (pixel_percentile_in_patch <= 0.15)
        # randMask2 = (pixel_percentile_in_patch > 0.15) & (pixel_percentile_in_patch <= 0.30)

        nullDeltaM0.append(np.nanmean(m0Null[highMask]) - np.nanmean(m0Null[lowMask]))
        nullDeltaM1.append(np.nanmean(m1Null[highMask]) - np.nanmean(m1Null[lowMask]))
        nullDeltaM2.append(np.nanmean(m2Null[highMask]) - np.nanmean(m2Null[lowMask]))

    nullDeltaM0 = np.array(nullDeltaM0) 
    nullDeltaM1 = np.array(nullDeltaM1) 
    nullDeltaM2 = np.array(nullDeltaM2) 

    print("Executing block bootstrap over patches.")
    bootDeltaM0, bootDeltaM1, bootDeltaM2 = [], [], []

    for trial in range(numTrials):
        sampledPatchesList = [] 
        for binID in range(len(latBinEdges)-1):
            binPatches = patchesByBin[binID]
            if len(binPatches) == 0:
                continue 
            sampledPatchesList.append(
                np.random.choice(binPatches, size=len(binPatches), replace=True)
            )
        sampledPatches = np.concatenate(sampledPatchesList) 
        bootIndices = np.concatenate(
            [patchIndicesMap[patchID] for patchID in sampledPatches]
        )

        groupingBoot = groupingValid[bootIndices]
        m0Boot = m0Valid[bootIndices]
        m1Boot = m1Valid[bootIndices] 
        m2Boot = m2Valid[bootIndices]

        # lowMaskBoot = groupingBoot <= q15Global 
        # highMaskBoot = groupingBoot >= q85Global

        binIDBoot = pixelBinID[bootIndices]
        lowMaskBoot = np.zeros(len(bootIndices), dtype=bool)
        highMaskBoot = np.zeros(len(bootIndices), dtype=bool)
        for binID, (q15Bin, q85Bin) in binThresholds.items():
            inBinBoot = binIDBoot == binID
            lowMaskBoot  |= inBinBoot & (groupingBoot <= q15Bin)
            highMaskBoot |= inBinBoot & (groupingBoot >= q85Bin)
        
        bootDeltaM0.append(np.nanmean(m0Boot[highMaskBoot]) - np.nanmean(m0Boot[lowMaskBoot]))
        bootDeltaM1.append(np.nanmean(m1Boot[highMaskBoot]) - np.nanmean(m1Boot[lowMaskBoot]))
        bootDeltaM2.append(np.nanmean(m2Boot[highMaskBoot]) - np.nanmean(m2Boot[lowMaskBoot]))

    bootDeltaM0 = np.array(bootDeltaM0) 
    bootDeltaM1 = np.array(bootDeltaM1) 
    bootDeltaM2 = np.array(bootDeltaM2) 

    # Statistical Evaluation
    validTrials0 = ~np.isnan(nullDeltaM0)
    validTrials1 = ~np.isnan(nullDeltaM1)
    validTrials2 = ~np.isnan(nullDeltaM2)

    pVal0 = (np.sum(np.abs(nullDeltaM0[validTrials0]) >= np.abs(obsDeltaM0)) + 1) / (validTrials0.sum() + 1)
    pVal1 = (np.sum(np.abs(nullDeltaM1[validTrials1]) >= np.abs(obsDeltaM1)) + 1) / (validTrials1.sum() + 1)
    pVal2 = (np.sum(np.abs(nullDeltaM2[validTrials2]) >= np.abs(obsDeltaM2)) + 1) / (validTrials2.sum() + 1)

    nullStd0 = np.nanstd(nullDeltaM0[validTrials0])
    nullStd1 = np.nanstd(nullDeltaM1[validTrials1])
    nullStd2 = np.nanstd(nullDeltaM2[validTrials2])

    zScore0 = obsDeltaM0 / nullStd0 if nullStd0 != 0 else np.nan
    zScore1 = obsDeltaM1 / nullStd1 if nullStd1 != 0 else np.nan
    zScore2 = obsDeltaM2 / nullStd2 if nullStd2 != 0 else np.nan

    # Generate Output Data Structure
    results = {
        "raw_arrays": {
            "null": {"M0": nullDeltaM0, "M1": nullDeltaM1, "M2": nullDeltaM2},
            "boot": {"M0": bootDeltaM0, "M1": bootDeltaM1, "M2": bootDeltaM2},
            "obs": {"M0": obsDeltaM0, "M1": obsDeltaM1, "M2": obsDeltaM2}
        }
    }

    # Generate Stats Table
    if generateStatsTable:
        stats_data = []
        for label, deltaVal, pVal, zScore, dataArr in [
            ("M0", obsDeltaM0, pVal0, zScore0, m0Valid),
            ("M1", obsDeltaM1, pVal1, zScore1, m1Valid),
            ("M2", obsDeltaM2, pVal2, zScore2, m2Valid),
        ]:
            stdHigh = np.nanstd(dataArr[highMask])
            stdLow = np.nanstd(dataArr[lowMask])
            fracPosHigh = np.nanmean(dataArr[highMask] > 0)
            fracPosLow = np.nanmean(dataArr[lowMask] > 0)
            
            stats_data.append({
                "Moment": label,
                "Observed_Delta": deltaVal,
                "p_value": pVal,
                "Effect_Size_Z": zScore,
                "Std_High": stdHigh,
                "Std_Low": stdLow,
                "Frac_Positive_High": fracPosHigh,
                "Frac_Positive_Low": fracPosLow
            })
            
        stats_df = pd.DataFrame(stats_data)
        results["stats_table"] = stats_df
        
        print("\n=== Statistics Table ===")
        print(stats_df.to_markdown(index=False, floatfmt=".4f"))
        print("========================\n")
    

    if makePlot:
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 8))
        binCount = 30
        
        plots_info = [
            (ax0, nullDeltaM0, bootDeltaM0, obsDeltaM0, "M0", "Moment 0", "gray", "green", "darkgreen"),
            (ax1, nullDeltaM1, bootDeltaM1, obsDeltaM1, "M1", "Moment 1", "gray", "teal", "darkslategray"),
            (ax2, nullDeltaM2, bootDeltaM2, obsDeltaM2, "M2", "Moment 2", "gray", "mediumpurple", "indigo")
        ]
        
        for ax, nullArr, bootArr, obsVal, sym, title, colNull, colBoot, colObs in plots_info:
            ax.hist(nullArr, bins=binCount, color=colNull, alpha=0.4, edgecolor="black", label="Null Distribution")
            ax.hist(bootArr, bins=binCount, color=colBoot, alpha=0.4, edgecolor="black", label="Observed Distribution (bootstrap)")
            ax.axvline(obsVal, color=colObs, linestyle="-", linewidth=2, label="True Observation (point estimate)")
            ax.axvline(np.nanmean(nullArr), color="crimson", linestyle="--", linewidth=2, label="Null Mean")
            
            ax.set_xlabel(f"$\\Delta$ {sym} (High - Low Dispersion)")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{title} Distribution Comparison")
            ax.legend()

        plt.tight_layout()
        plt.show()

    return results
