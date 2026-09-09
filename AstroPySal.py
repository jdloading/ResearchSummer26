import numpy as np 
import healpy as hp 
import libpysal 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import sparse 
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import shortest_path 
from esda.moran import Moran_Local_BV 

class AstroPySal:
    """
    Spatial analysis tools for astrophysica HEALPix data using PySal.
    """
    def __init__(
        self, 
        nside,
        keepMask=None,
        dropIslands=True,
        alpha=0.05,
        permutations=999
    ):
        self.nside = nside
        self.dropIslands = dropIslands
        self.alpha = alpha 
        self.permutations = permutations

        npix = hp.nside2npix(nside) 

        if keepMask is None: 
            self.keepMask = np.ones(npix, dtype=bool) 
        else: 
            self.keepMask = keepMask.copy()

        self.keepIds = None 
        self.w = None

        self.lisaCmap = ListedColormap(
        [
        "#e0e0e0",  # Not significant (Neutral light gray)
        "#E50000",  # HH: Bright Red
        "#FFA500",  # LH: Soft Pink/Light Coral
        "#0343DF",  # LL: Dark Blue
        "#029386",  # HL: Deep Purple 
        ]
        )

        self.lisaNorm = BoundaryNorm(
            [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
            self.lisaCmap.N,
        )

        self.lisaLabels = [
            "Not Significant", 
            "HH (High-High)", 
            "LH (Low-High)",
            "LL (Low-Low)",
            "HL (High-Low)",
        ]
        
    def build_healpix_weights(self):
        """
        Build row-standardized spatial weights for the retained HEALPix pixels. 

        Returns
        --------
        libpysal.weights.W
            Spatial weights object.
        """

        npix = hp.nside2npix(self.nside) 
        keepMask = self.keepMask.copy()

        while True:
            keepIds = np.where(keepMask)[0]
            nKeep = len(keepIds)

            idMap = np.full(npix, -1, dtype=np.int64)
            idMap[keepIds] = np.arange(nKeep)

            allNeighbors = hp.get_all_neighbours(self.nside, keepIds)
            rowIndices = np.repeat(np.arange(nKeep), 8)
            colIndicesOld = allNeighbors.T.flatten()

            validMask = colIndicesOld != -1
            rowIndices = rowIndices[validMask]
            colIndicesOld = colIndicesOld[validMask]

            colIndices = idMap[colIndicesOld]
            keepEdge = colIndices != -1 

            rowIndices = rowIndices[keepEdge]
            colIndices = colIndices[keepEdge]

            degree = np.zeros(nKeep, dtype=int)
            np.add.at(degree, rowIndices, 1)

            islandMask = degree == 0 

            if not self.dropIslands or not np.any(islandMask):
                break 

            keepMask[keepIds[islandMask]]=False

        data = np.ones(len(rowIndices), dtype=float)

        adjMatrix = sparse.coo_matrix(
            (data, (rowIndices, colIndices)),
            shape=(nKeep, nKeep),
            
        )
        w = libpysal.weights.W.from_sparse(adjMatrix) 
        w.transform="R"

        assert w.n == nKeep, (
            f"internal error: w.n={w.n} != nKeep={nKeep}"
        )

        self.keepMask = keepMask 
        self.keepIds = keepIds 
        self.w = w 

        return w 

    def compute_local_means(
        self, 
        dataMap, 
        radiusDeg=5.0,
        minValidFraction=0.5,
    ):
        """
        Compute a circular local mean for each HEALPix pixel. 
        
        Parameters
        ----------
        dataMap: array-like 
            full-sky HEALPix map at self.nside. Invalid pixels represented as hp.UNSEEN or nan values.
            
        radiusDeg: float 
            angular radius of the local window in degrees. 

        minValidFraction : float 
            Minimum fraction of pixels in a local window that must contain valid data for a local mean to 
            be calculated. 
        Returns:
        _________
        LocalMean: numpy.ndarray
            Full-sky HEALPix array containing the local mean for each pixel. 
            pixels without sufficient valid data are set to hp.UNSEEN 
            
        """
        dataMap = np.asarray(dataMap, dtype=np.float64)

        npix = hp.nside2npix(self.nside)

        if len(dataMap) != npix:
            raise ValueError(
                f"dataMap has {len(dataMap)} pixels, "
                f"but nside ={self.nside} require {npix} pixels."
            )
        if radiusDeg <= 0:
            raise ValueError("radiusDeg must be greater than zero.")

        if not 0 < minValidFraction <= 1:
            raise ValueError(
                "minValidFraction must be greater than 0 and less than or equal to 1."
            )
        validMask = (
            np.isfinite(dataMap) 
            & (dataMap != hp.UNSEEN)
            )
        localMean = np.full(npix, hp.UNSEEN, dtype=np.float64) 
        radiusRad = np.radians(radiusDeg) 
        validIds  = np.where(validMask)[0]

        for pixId in validIds: 

            centerVec = hp.pix2vec(
                self.nside, 
                pixId, 
            )

            windowPixIds = hp.query_disc(
                self.nside, 
                centerVec, 
                radiusRad, 
                inclusive=True
            )
            windowValidMask = validMask[windowPixIds]

            nValid = np.sum(windowValidMask)
            nTotal = len(windowPixIds) 

            validFraction = nValid / nTotal 

            if validFraction < minValidFraction: 
                continue 

            localMean[pixId] = np.mean(
                dataMap[windowPixIds[windowValidMask]]
            )
        return localMean

    def local_lisa_classification(
        self, 
        xArray, 
        yArray, 
        radiusDeg=5.0,
        minValidFraction=0.5
    ):
        """Classify bivariate HEALPix pixels using locally normalized means. 
        
        The central X pixel is classified relative to its local X mean. 
        The neighboring Y field is classified relative to the local Y mean 
        at each neighboring pixel. 
        
        Quadrant codes:_
            1 = HH 
            2 = LH 
            3 = LL_
            4 = HL

        Pixels without sufficient local data are set to hp.UNSEEN. 

        Returns 
        -------
        dict 
            Local LISA classification results and intermediate maps. 
            
        """
        if self.w is None or self.keepIds is None:_
        self.build_healpix_weights() 

        xArray = np.asarray(xArray, dtype=np.float64)
        yArray = np.asarray(yArray, dtype=np.float64)

        npix   = hp.nside2npix(self.nside) 

        if len(xArray) != npix: 
            raise ValueError(
                f"xArray has {len(xArray)} pixels, "
                f"but nside={self.nside} requires {npix} pixels."
            )
        if len(yArray) != npix:
            raise ValueError(
                f"yArray has {len(yArray)} pixels,  "
                
            )
        localMeanX = self.compute_local_means(
            xArray, 
            radiusDeg=radiusDeg, 
            minValidFraction=minValidFraction,
        )

        localMeanY = self.compute_local_means(
            yArray, 
            radiusDeg = radiusDeg, 
            minValidFraction=minValidFraction,
        )
        validX = (
            np.isfinite(xArray)
            & (xArray != hp.UNSEEN)
            & np.isfinite(localMeanX)
            & (localMeanX != hp.UNSEEN)
        )

        validY = (
            np.isfinite(yArray)
            & (yArray != hp.UNSEEN)
            & np.isfinite(localMeanY)
            & (localmeanY != hp.UNSEEN)
        )
        validBoth = validX & validY

        xResidual = np.full(
            npix, 
            hp.UNSEEN, 
            dtype=np.float64,
        )

        yResidual = np.full(
            npix, 
            hp.UNSEEN, 
            dtype = np.float64,
        )
        localNeighborY = np.full(
            npix, 
            hp.UNSEEN, 
            dtype=np.float64,
        )
        validIds = self.keepids[validBoth[self.keepIds]]

        localIndices = np.where(
            validBoth[self.keepIds]
        )[0]

        if len(validIds) == 0:
            raise ValueError(
                "No pixels remain after applying local validity requirements."
            )

        localSparse = self.w.sparse[
            localIndices
        ][:, localIndices]

        localW = libpysal.weights.W.from_sparse(
            localSparse
        )

        localW.transform = "R"

        yValid = yResidual[validIds]

        neighborY = localW.sparse @ yValid 

        localNeighborY[validIds] = np.asarray(
            neighborY
        ).ravel()

        localMap = np.full(
            npix, 
            hp.UNSEEN, 
            dtype=np.float64,
        )

        xHigh = xResidual > 0 
        xLow = xResidual < 0 

        yHigh = localNeighborY > 0 
        yLow  = localNeighborY < 0 

        hhMask = validBoth & xHigh & yHigh 
        lhMask = validBoth & xLow & yHigh 
        llMask = validBoth & xLow & yLow 
        hlMask = validBoth & xHigh & yLow 

        localMap[hhMask] = 1 
        localMap[lhMask] = 2
        localMap[llMask] = 3
        localMap[hlMask] = 4

        return {
            "localMap": localMap,
            "localMeanX": localMeanX,
            "localMeanY": localMeanY,
            "xResidual": xResidual,
            "yResidual": yResidual,
            "localNeighborY": localNeighborY,
            "localKeepIds": validIds,
            "localW": localW,
            "radiusDeg": radiusDeg,
            "minValidFraction": minValidFraction,
        }
        
    def lisa_bv_local_report(
        self, 
        xArray, 
        yArray, 
        label, 
        radiusDeg=2.0,
        minValidFraction=0.5,
        permutations=999
    ):
        """Calculate a locally normalized bivariate LISA_statistic. 
        
        Each map is locally mean-subtracted before calculating Moran's I. 
        The local mean is calculated within an angular radius around each pixel. Pixels without 
        sufficient data are excluded.
        """
        xArray = np.asarray(xArray, dtype=np.float64)
        yArray = np.asarray(yArray, dtype=np.float64)

        npix = hp.nside2npix(self.nside)

        if len(xArray) != npix:
            raise ValueError(
                f"xArray has {len(xArray)} pixels, "
                f"but nside = {self.nside} requires {npix} pixels."
            )
        if len(yArray) != npix:
            raise ValueError(
                f"yArray has {len(yArray)} pixels, "
                f"but nside={self.nside} requires {npix} pixels."
            )

        localMeanX = self.compute_local_means(
                xArray, 
                radiusDeg=radiusDeg, 
                minValidFraction=minValidFraction,
        )
        localMeanY = self.compute_local_means(
                yArray,
                radiusDeg=radiusDeg,
                minValidFraction=minValidFraction,
        )
        xResidual = np.full(
                npix, 
                hp.UNSEEN,
                dtype=np.float64,
        )
        yResidual = np.full(
                npix, 
                hp.UNSEEN, 
                dtype=np.float64,
        )
        validX = (
                np.isfinite(xArray)
                & (xArray!=hp.UNSEEN)
                & (np.isfinite(localMeanX))
                & (localMeanX != hp.UNSEEN)
        )
        validY = (
                np.isfinite(yArray)
                & (yArray != hp.UNSEEN)
                & (np.isfinite(localMeanY))
                & (localMeanY != hp.UNSEEN)
        )
        validBoth = validX & validY
        xResidual[validBoth]= (
            xArray[validBoth] - localMeanX[validBoth]
            )
        yResidual[validBoth] = (
                yArray[validBoth] - localMeanY[validBoth]
            )
        localKeepIds = self.keepIds[
                validBoth[self.keepIds]
            ]
        localIndices = np.where(
                validBoth[self.keepIds]
            )[0]
        if len(localKeepIds) == 0:
            raise ValueError(
                "No pixels remain after applying the local validity requirements."
            )
        localSparse = self.w.sparse[
            localIndices
        ][:,localIndices]
        localW = libpysal.weights.W.from_sparse(
            localSparse
        )
        localW.transform="R"

        xValid = xResidual[localKeepIds]
        yValid = yResidual[localKeepIds]

        lisaBV = Moran_Local_BV(
            xValid, 
            yValid,
            localW,
            permutations=permutations,
        )

        lisaBV.significanceMethod = "pysal"
        lisaBV.localMeanX = localMeanX
        lisaBV.localMeanY = localMeanY
        lisaBV.xResidual = xResidual
        lisaBV.yResidual = yResidual
        lisaBV.localKeepIds = localKeepIds
        lisaBV.localW = localW
        lisaBV.radiusDeg = radiusDeg
        lisaBV.minValidFraction = minValidFraction

        sig = lisaBV.p_sim < self.alpha

        counts = {
            q: int(np.sum(sig & (lisaBV.q == q)))
            for q in [1, 2, 3, 4]
        }

        print(
            f"{label:20s} " 
            f"significant={np.sum(sig):6d}/{len(xValid)}  "
            f"HH={counts[1]:5d}  "
            f"LH={counts[2]:5d}  "
            f"LL={counts[3]:5d}  "
            f"HL={counts[4]:5d}"
        )

        return (
            lisaBV, 
            localKeepIds, 
            localMeanX, 
            localMeanY, 
            xResidual, 
            yResidual,
        )
    def generate_gaussian_surrogates(
        self,
        dataMap, 
        nSurrogates=100,
        lmax=None,
    ):
        """
        Generate Guassian HEALPix surrogate maps taht preserce the angular power spectrum of the 
        input map 
        """

        if lmax is None: 
            lmax = 3 * self.nside - 1 

        workingMap = dataMap.copy()

        validMask = (
            np.isfinite(workingMap)
            & (workingMap != hp.UNSEEN)
        )

        fillValue = np.nanmean(workingMap[validMask])
        workingMap[~validMask] = fillValue 

        cl = hp.anafast(
            workingMap, 
            lmax=lmax, 
        )

        surrogates = []

        for i in range(nSurrogates): 

            surrogate=hp.synfast(
                cl, 
                nside=self.nside, 
                lmax=lmax, 
                #verbose=False,
            )
            surrogate[~validMask] = hp.UNSEEN
            surrogates.append(surrogate) 
        return surrogates 

    def _run_surrogate_lisa(
        self,
        xArray,
        surrogateMap,
        quadrants=(1, 4),
    ):
        """
        Run a single surrogate LISA realization and extract the resulting
        significant regions.
    
        Returns
        -------
        lisa : Moran_Local_BV
            LISA object for the surrogate realization.
    
        regions : list
            Output from extract_lisa_bv_regions().
        """
    
        xValid = xArray[self.keepIds].astype(np.float64)
        surrogateValid = surrogateMap[self.keepIds].astype(np.float64)

        lisa = self.lisa_bv_report(
            xArray,
            surrogateMap,
            significanceMethod="pysal",
        )

        regions = self.extract_lisa_bv_regions(
            lisa,
            quadrants=quadrants,
            alpha=self.alpha,
            printRes=False,
        )
        
        return lisa, regions
    
    def compare_regions_to_surrogates(
        self, 
        xArray, 
        realRegions,
        surrogateMaps, 
        quadrants=(1,4), 
        alpha=None, 
        printResults=True,
    ):
        """
        Compare real LISA region statistics against surrogate realizations. Serves as a regional 
        significance test, as opposed to the pixel level significance that the built in pySal function   does"""
        xValid = xArray[self.keepIds].astype(np.float64) 

        largestRegions    = []
        numberRegions     = []
        meanRegionSizes   = []
        medianRegionSizes = []
        totalPixels       = []
        largestFractions  = [] 

        for surrogate in surrogateMaps: 
            
            lisa, regions = self._run_surrogate_lisa(
                xArray,
                surrogate,
                quadrants=quadrants,
            )
            
            if len(regions) == 0: 
                largestRegions.append(0)
                numberRegions.append(0) 
                meanRegionSizes.append(0)
                medianRegionSizes.append(0)
                totalPixels.append(0)
                largestFractions.append(0) 
                continue 

            regionSizes = np.array(
                [region["size"] for region in regions] 
            )

            largestRegions.append(regionSizes.max())
            numberRegions.append(len(regionSizes))
            meanRegionSizes.append(regionSizes.mean())
            medianRegionSizes.append(np.median(regionSizes))
            totalPixels.append(regionSizes.sum())
            largestFractions.append(regionSizes.max() / regionSizes.sum())

        if len(realRegions) == 0: 
            realSizes = np.array([0])
        else: 
            realSizes = np.array(
                [region["size"] for region in realRegions]
            ) 

        realStatistics = {
            "largestRegion": realSizes.max(), 
            "numberRegions": len(realSizes) if len(realRegions) > 0 else 0, 
            "meanRegionSize": realSizes.mean(), 
            "medianRegionSize": np.median(realSizes), 
            "totalPixels": realSizes.sum(), 
            "largestFraction": realSizes.max() / realSizes.sum() 
            if realSizes.sum() > 0 else 0, 
        }

        surrogateStatistics = {
            "largestRegion": np.array(largestRegions), 
            "numberRegions": np.array(numberRegions), 
            "meanRegionSize": np.array(meanRegionSizes), 
            "medianRegionSize": np.array(medianRegionSizes), 
            "totalPixels": np.array(totalPixels),
            "largestFraction": np.array(largestFractions),
        }

        results  = {} 

        for statistic in surrogateStatistics: 

            distribution = surrogateStatistics[statistic]

            exceedances = np.sum(
                distribution >= realStatistics[statistic]
            )

            pValue = (
                exceedances + 1
            ) / (
                len(distribution) + 1
            )

            results[statistic] = {
                "real": realStatistics[statistic], 
                "mean": distribution.mean(),
                "std": distribution.std(),
                "p": pValue, 
                "distribution": distribution, 
            }

        return results 

    def compare_quadrant_abundance(
        self, 
        xArray, 
        yArray, 
        surrogateMaps,
    ):
        """
        Compare abundance of sig. HH, LH, LL and HL pixels against Gaussian surrogate realizations. 
        """

        xValid = xArray[self.keepIds].astype(np.float64)
        yValid = yArray[self.keepIds].astype(np.float64)

        realLisa = Moran_Local_BV(
            xValid,
            yValid, 
            self.w,
            permutations=self.permutations,
        )

        realSig = realLisa.p_sim < self.alpha 

        realCounts = {}

        for quadrant in [1, 2, 3, 4]: 
            realCounts[quadrant] = np.sum(
                realSig & 
                (realLisa.q == quadrant)
            )

        surrogateCounts = {
            1: [],
            2: [],
            3: [],
            4: [],
        }

        for surrogate in surrogateMaps: 

            lisa, _ = self._run_surrogate_lisa(
                xArray,
                surrogate,
            )

            sig = lisa.p_sim < self.alpha 

            for quadrant in [1,2,3,4]: 
                surrogateCounts[quadrant].append(
                    np.sum(
                        sig &
                        (lisa.q == quadrant)
                    )
                )
        quadrantNames = {
            1: "HH",  
            2: "LH",
            3: "LL",
            4: "HL",
        }

        results = {}

        for quadrant in [1,2,3,4]:
            distribution = np.asarray(
                surrogateCounts[quadrant]
            )

            exceedances = np.sum(
                distribution >= realCounts[quadrant]
            )

            pValue = (
                exceedances + 1
            ) / (
                len(distribution) + 1
            )

            results[quadrantNames[quadrant]] = {
                "real": int(realCounts[quadrant]),
                "mean": distribution.mean(), 
                "std": distribution.std(),
                "p": pValue,
                "distribution": distribution,
                
            }
        return results

    def report_quadrant_abundance(
        self,
        results,
    ):
        """
        Print the quadrant abundance comparison.
        """
    
        print("=" * 60)
        print("Quadrant Abundance Test")
        print("=" * 60)
    
        for quadrant in ["HH", "LH", "LL", "HL"]:
    
            r = results[quadrant]
    
            print(quadrant)
    
            print(f"  Real Count:        {r['real']}")
    
            print(f"  Surrogate Mean:    {r['mean']:.1f}")
    
            print(f"  Surrogate Std:     {r['std']:.1f}")
    
            print(f"  Empirical p:       {r['p']:.4f}")
    
            if r["p"] < self.alpha:
    
                print("  Result: Significant")
    
            else:
    
                print("  Result: Not significant")
    
            print("-" * 60)

    def compare_region_hierarchy(
        self, 
        xArray, 
        realRegions, 
        surrogateMaps, 
        quadrants=(1,4), 
        nRanks=10,
    ):
        """
        Compare to the hierarchy of the largest regions against Gaussian surrogate realizations
        """

        xValid = xArray[self.keepIds].astype(np.float64) 

        realSizes = np.sort(
            [r["size"] for r in realRegions]
            
        )[::-1]

        realHierarchy = np.zeros(nRanks) 

        realHierarchy[:min(len(realSizes), nRanks)] = \
            realSizes[:nRanks]

        surrogateHierarchy = np.zeros(

            (
                len(surrogateMaps),
                nRanks,
            )
        )
        for i, surrogate in enumerate(surrogateMaps): 

            _, regions = self._run_surrogate_lisa(
                xArray,
                surrogate,
                quadrants=quadrants,
            )
            
            if len(regions)==0:
                continue 
            sizes = np.sort(
                [r["size"] for r in regions]
            )[::-1]
            surrogateHierarchy[
                i, 
                :min(
                    len(sizes),
                    nRanks,
                )
            ] = sizes[:nRanks]
        results = []

        for rank in range(nRanks): 
            distribution = surrogateHierarchy[:, rank]
            exceedances = np.sum(
                distribution >= realHierarchy[rank]
            )

            pValue = (
                exceedances + 1
            ) / (
                len(distribution) + 1
            )
            results.append(
                {
                    "rank": rank + 1,
    
                    "real": realHierarchy[rank],
    
                    "mean": distribution.mean(),
    
                    "std": distribution.std(),
    
                    "p": pValue,
    
                    "distribution": distribution,
                }
            )

        return results 

    def report_region_hierarchy(
        self,
        results,
    ):
        """
        Print the ranked region comparison.
        """
    
        print("=" * 72)
        print("Region Hierarchy Test")
        print("=" * 72)
    
        print(
            f"{'Rank':>6}"
            f"{'Real':>12}"
            f"{'Mean':>12}"
            f"{'Std':>12}"
            f"{'p':>12}"
        )
    
        print("-" * 72)
    
        for r in results:
    
            print(
    
                f"{r['rank']:6d}"
    
                f"{r['real']:12.0f}"
    
                f"{r['mean']:12.1f}"
    
                f"{r['std']:12.1f}"
    
                f"{r['p']:12.4f}"
    
            )

    
    
    def report_surrogate_statistics(
        self, 
        results,
    ):
        """
        Print a summary of surrogate region statistics. 
        """

        print("=" * 60)
        print("Region Significance Test")
        print("=" * 60) 

        statisticNames = {
            "largestRegion": "Largest Region", 
            "numberRegions": "Number of Regions",
            "meanRegionSize": "Mean Region Size", 
            "medianRegionSize": "Median Region Size",
            "totalPixels": "Total Significant Pixels",
            "largestFraction": "Largest Region Fraction",
        }

        for key, title in statisticNames.items():
            statistic = results[key] 

            print(title)
            print(f" Real:            {statistic['real']:.3f}")
            print(f" Surrogate Mean:  {statistic['mean']:.3f}")
            print(f" Surrogate Std:   {statistic['std']:.3f}")
            print(f" Empirical P:     {statistic['p']:.4f}")

            if statistic["p"] < self.alpha: 
                print("Result significant") 
            else: 
                print("Result insignificant")
            print("-"*60)

    
    def calculate_gaussian_significance(
        self, 
        xArray, 
        yArray, 
        nSurrogates=100,
    ):
        """
        Compute LISA significance by comparing the observed local Moran's I values against Gaussian 
        surrogate realizations of the neighbour-producing field.
        """

        xValid = xArray[self.keepIds].astype(np.float64)
        yValid = yArray[self.keepIds].astype(np.float64)

        realLisa = Moran_Local_BV(
            xValid, 
            yValid, 
            self.w,
            permutations=0,
        )

        surrogateMaps = self.generate_gaussian_surrogates(
            yArray,
            nSurrogates=nSurrogates,
        )
        surrogateI = np.zeros(
            (
                nSurrogates,
                len(xValid),
            )
        )

        for i, surrogate in enumerate(surrogateMaps): 
            surrogateValid = surrogate[self.keepIds]
            surrogateLisa = Moran_Local_BV(
                xValid,
                surrogateValid, 
                self.w, 
                permutations=0,
            )

            surrogateI[i] = surrogateLisa.Is

        pValues = np.mean(
            np.abs(surrogateI)
            >= np.abs(realLisa.Is),
            axis=0, 
        )

        realLisa.p_sim=pValues 
        realLisa.significanceMethod="gaussian"
        
        return realLisa 
    
    def lisa_bv_report(
        self, 
        xArray,
        yArray, 
        label="", 
        permutations=None,
        significanceMethod="pysal",
        nSurrogates=100,
    ):
        """
        Runs a bivariate Local Moran's I analysis and reports significant quadrant counts. 

        Returns
        -------
        esda.moran.Moran_Local_BV 
            LISA-BV result object. 
        """

        if self.w is None or self.keepIds is None: 
            self.build_healpix_weights()

        if permutations is None: 
            permutations = self.permutations

        xValid = xArray[self.keepIds].astype(np.float64) 
        yValid = yArray[self.keepIds].astype(np.float64) 

        if significanceMethod == "pysal":

            xValid = xArray[self.keepIds].astype(np.float64)
            yValid = yArray[self.keepIds].astype(np.float64)

            lisaBV = Moran_Local_BV(
                xValid, 
                yValid, 
                self.w, 
                permutations=self.permutations, 
            )

            lisaBV.significanceMethod = "pysal"

        elif significanceMethod == "gaussian": 

            lisaBV = self.calculate_gaussian_significance(
                xArray, 
                yArray, 
                nSurrogates=nSurrogates, 
            )
        else: 

            raise ValueError(
                f"Unknown significance method: {significanceMethod}"
            )

        sig = lisaBV.p_sim < self.alpha 

        counts = {
            q: int(np.sum(sig & (lisaBV.q == q)))
            for q in [1,2,3,4]
        }
        print(
            f"{label:20s} significant={np.sum(sig):6d}/{len(xValid)}   "
            f"HH={counts[1]:5d}  LH={counts[2]:5d}  "
            f"LL={counts[3]:5d}  HL={counts[4]:5d}"
        )

        return lisaBV 

    def extract_lisa_bv_regions(
        self,
        lisaBV, 
        alpha=None, 
        quadrants=(1,2,3,4),
        printRes=False,
    ):
        """
        Find contiguous spatial clusters o significant LISA-BV pixels, report each region's quadrant composition, and 
        calculate a spherical-mean center for each region in GAL coords (l,b): 

        QUADRANT CODES: 
        1 = HH
        2 = LH 
        3 = LL
        4 = HL
        """

        if self.w is None or self.keepIds is None: 
            raise RuntimeError(
                "Spatial weights have not been built."
                "Call build_healpix_weights() first."
            )

        if alpha is None: 
            alpha = self.alpha

        quadrantNames = {
            1: "HH",
            2: "LH", 
            3: "LL", 
            4: "HL", 
        }

        sig = (
            (lisaBV.p_sim < alpha)
            & np.isin(lisaBV.q, quadrants)
        )

        sigLocalIdx = np.where(sig)[0]
        sigQuadCodes = lisaBV.q[sigLocalIdx]

        subAdj = self.w.sparse[sigLocalIdx][:, sigLocalIdx]

        nComponents, labels = connected_components(
            subAdj, 
            directed=False,
        )

        regionPixIds = self.keepIds[sigLocalIdx]

        results=[]

        for regionId in range(nComponents): 
            mask = labels == regionId
            pixIds = regionPixIds[mask]
            localIndices = sigLocalIdx[mask]
            quadCodes = sigQuadCodes[mask]

            vecs = np.array(
                hp.pix2vec(
                    self.nside, 
                    pixIds,
                )
            ).T

            meanVec = vecs.mean(axis=0)
            meanVec /= np.linalg.norm(meanVec)

            theta, phi = hp.vec2ang(meanVec) 

            centerLat = 90.0 - np.degrees(theta) 
            centerLon = np.degrees(phi) 

            if centerLon > 180: 
                centerLon -= 360 

            angularDistances = np.degrees(
                np.arccos(
                    np.clip(
                        vecs @ meanVec, 
                        -1, 
                        1,
                    )
                )
            )
            
            counts = {
                quadrantNames[q]: int(np.sum(quadCodes == q))
                for q in quadrants 
            } 

            presentTypes = [
                t for t, c in counts.items()
                if c > 0 
            ]

            if len(presentTypes) == 1: 
                regionType = presentTypes[0]
            else: 
                dominant = max(counts, key=counts.get) 
                regionType = (
                    f"mixed (dominant {dominant}: {counts})"
                )
            vecs = np.array(
                hp.pix2vec(self.nside, pixIds)
            ).T

            meanVec = vecs.mean(axis=0)
            meanVec /= np.linalg.norm(meanVec)

            theta, phi = hp.vec2ang(meanVec) 

            centerLat = 90.0 - np.degrees(theta[0])
            centerLon = np.degrees(phi[0])

            if centerLon > 180: 
                centerLon -= 360 

            results.append(
                {
                    "regionId": regionId, 
                    "size": len(pixIds), 
                    "type": regionType, 
                    "composition": counts, 
                    "centerGalLat": centerLat, 
                    "centerGalLon": centerLon,
                    "pixIds": pixIds,
                    "centerVec": meanVec, 
                    "angularDistances": angularDistances, 
                    "pixIds": pixIds, 
                    "localIndices": localIndices, 
                }
            )
        results.sort(
            key=lambda region: -region["size"]
        )

        if printRes: 
            for region in results[:15]:
                print(
                    f"region {region['regionId']:3d}  "
                    f"size={region['size']:5d}        "
                    f"type={region['type']:25s}"
                    f"\ncenter=("
                    f"l={region['centerGalLon']:7.2f}, "
                    f"b={region['centerGalLat']:7.2f})"
                    f"\n"
                    f"----------------------------------------------------------------"
                )

        return results 

    def swap_lisa_bv_quadrants(
    self,
    lisaMap,
    ):
        """
        Return a copy of a LISA cluster map with the Low-High and
        High-Low quadrants exchanged. This is useful when plotting
        the reversed variable ordering in a bivariate Moran analysis.
        """

        swappedMap = lisaMap.copy()

        lhMask = lisaMap == 2
        hlMask = lisaMap == 4

        swappedMap[lhMask] = 4
        swappedMap[hlMask] = 2

        return swappedMap
    
    def build_lisa_bv_map(self, lisaBV, alpha=None):
        """
        Maps Moran_Local_BV results onto a full-sky HEALPix array. 
        """
        if alpha is None: 
            alpha = self.alpha 
        npix = hp.nside2npix(self.nside) 
        quadMap = np.full(npix, hp.UNSEEN)
        sig = lisaBV.p_sim < alpha 
        code = np.where(sig, lisaBV.q, 0)
        quadMap[self.keepIds] = code 
        return quadMap
    def plot_lisa_bv_map(
    self,
    lisaMap,
    title="LISA Bivariate Cluster Map",
    reverse=False,
    showLegend=True,
    ):
        """
        Plot a LISA-BV cluster map using the built-in categorical colormap.
        """

        if reverse:
            lisaMap = self.swap_lisa_bv_quadrants(lisaMap)

        hp.newvisufunc.projview(
            lisaMap,
            cmap=self.lisaCmap,
            norm=self.lisaNorm,
            min=0,
            max=4,
            coord=["G"],
            graticule=True,
            graticule_labels=True,
            title=title,
            cbar=False,
        )

        if showLegend:

            labels = self.lisaLabels.copy()

            if reverse:
                labels[2], labels[4] = labels[4], labels[2]

            colors = self.lisaCmap.colors

            handles = [
                plt.Rectangle((0, 0), 1, 1, color=color)
                for color in colors
            ]

            plt.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.30),
                ncol=3,
                frameon=True,
            )
     
    def plot_region(
        self, 
        region
    ):
        pixIds = region["pixIds"]

        npix = hp.nside2npix(self.nside)
        highlightMap = np.full(npix, hp.UNSEEN)

        highlightMap[pixIds] = 1

        highlightMap[highlightMap < -1e30] = 0
        
        hp.newvisufunc.projview(
        highlightMap,
        coord=["G"],
        graticule=True,
        graticule_labels=True,
        title=f"Region {region['regionId']} ({len(pixIds)} pixels)",
        cbar=False,
        cmap='plasma',
        min = np.nanpercentile(highlightMap, 0),
        max = np.nanpercentile(highlightMap, 100)
        )
    

    def describe_region(
        self, 
        regionId, 
        regionPixIds, 
        regionLabels,
    ):
        """Return the HEALPix pixel IDs and galactic coordinates for a region, 
        along with a printed angular extent summary.
        """

        mask = regionLabels == regionId
        pixIds = regionPixIds[mask]

        theta, phi = hp.pix2ang(
            self.nside, 
            pixIds,
        )

        galLatRegion = 90.0 - np.degrees(theta) 
        galLonRegion = np.degrees(phi) 

        galLonRegion = np.where(
            galLonRegion > 180, 
            galLonRegion - 360, 
            galLonRegion,
        )

        print(
            f" galactic latitude:   "
            f"{galLatRegion.min():.1f} to "
            f"{galLatRegion.max():.1f} deg " 
            f"(center ~{galLatRegion.mean():.1f})"
        )

        return (
            pixIds, 
            galLatRegion, 
            galLonRegion, 
        )

        print(f"Region {regionId}: {len(pixIds)} pixels")

    def compute_region_morphology(
        self, 
        region, 
    ):
        """
        Compute basic morphology statistics for a single region. 
        Paramaters
        ----------
        region: dict 
            output from extract_lisa_bv__regions()
            
        returns 
        -------
        dict 
            Morphological properties of the region
        """

        pixIds = region["pixIds"]

        centerVec = region["centerVec"] 

        angularDistances = region["angularDistances"] 

        extentDeg = angularDistances.max()

        meanRadius = angularDistances.mean()

        radiusStd = angularDistances.std()

        pixelArea = hp.nside2pixarea(
            self.nside, 
            degrees=True,
        )

        regionArea = len(pixIds) * pixelArea 

        circleArea = np.pi * extentDeg ** 2 

        filllingFactor = regionArea / circleArea 

        localIndices = region["localIndices"] 

        subgraph = self.w.sparse[
            localIndices
        ][:, localIndices] 

        graphDistances = shortest_path(
            subgraph, 
            directed=False, 
            unweighted=True,
        )

        finiteDistances = graphDistances[
            np.isfinite(graphDistances)
        ]
        graphDiameter = finiteDistances.max()

        meanDegree = np.asarray(
            subgraph.sum(axis=1)
        ).mean()
        
        return{
            "extentDeg": extentDeg, 
            "meanRadius": meanRadius, 
            "radiusStd": radiusStd,
            "fillingFactor": fillingFactor, 
            "graphDiameter": graphDiameter, 
            "meanDegree": meanDegree, 
        }

    def compare_region_persistence(
        self,
        xArray,
        yArray,
        quadrants=(1,4), 
        alphas=None,
    ):
        """
        Measure how the detected LISA regoins persist as the PySal significance threshold is varied.
        
        params: 
        -------
        xArray: ndarray
            primary data array 
        
        yArray: ndarray
            neighbour-producing data array 
        
        quadrants: tuple, optional
            LISA quadrants to include 
            
        alphas: sequence, optional 
            Significance thresholds to test. 
            defaults to [0.20, 0.10, 0.05, 0.02, 0.01].
            
        Returns
        -------
        dict 
            Region statistics as a function of alpha. 
        """
        if alphas is None: 
            alphas = [
                0.20,
                0.10,
                0.05,
                0.02,
                0.01
            ]

        xValid = xArray[self.keepIds].astype(np.float64)
        yValid = yArray[self.keepIds].astype(np.float64)

        lisa = Moran_Local_BV(
            xValid,
            yValid, 
            self.w,
            permutations=self.permutations,
        )

        largestRegion = [] 

        numberRegions = []

        totalPixels = [] 

        for alpha in alphas: 
            regions = self.extract_lisa_bv_regions(
                lisa, 
                quadrants=quadrants, 
                alpha=alpha,
                printRes=False,
            )
            numberRegions.append(
                len(regions)
            )
            if len(regions) == 0: 
                largestRegions.append(0)
                totalPixels.append(0)
                continue 
            sizes = np.array(
                [
                    region["size"]
                    for region in regions
                ]
            )
            largestRegion.append(
                sizes.max()
            )
            totalPixels.append(
                sizes.sum()
            )
        return{
            "alpha": np.asarray(alphas),
            "largestRegion":np.asarray(
                largestRegion
            ),
            "numberRegions": np.asarray(
                numberRegions
            ),
            "totalPixels": np.asarray(
                totalPixels
            ),
        }

    def report_region_persistence(
        self, 
        persistance,
    ):
        """
        Print a summary of region persistence across significance thresholds. 
        """

        print("=" * 70) 
        print("Region persistence")
        print("=" * 70) 

        print(
            f"{'Alpha':>8}"
            f"{'Largest':>12}"
            f"{'Regions':>12}"
            f"{'Pixels':>12}"
        )

        print("-" * 70)

        for alpha, largest, nRegions, pixels in zip(
            persistence["alpha"], 
            persistence["largestRegion"],
            persistence["numberRegions"],
            persistence["totalPixels"],
        ):
            print(
                f"{alpha:8.2f}"
                f"{largest:12d}"
                f"{nRegions:12d}"
                f"{pixels:12d}"
            )

        
    def plot_region_cutout(
        self, 
        dataMap, 
        region, 
        paddingFactor=2.0,
        title=None, 
        unit="",
        cmap=None,
    ):
        """
        Plot a gnomonic projection centered on a region's spherical mean position.
        """

        centerVec = hp.ang2vec(
            region["centerGalLon"], 
            region["centerGalLat"], 
            lonlat=True,
        )

        pixVecs = np.array(
            hp.pix2vec(
                self.nside, 
                region["pixIds"], 
            )
        ).T

        angDist = np.degrees(
            np.arccos(
                np.clip(
                    pixVecs @ centerVec, 
                    -1, 
                    1,
                )
            )
        )

        radiusDeg = angDist.max() * paddingFactor

        resoArcmin = max(
            radiusDeg * 60 / 100, 
            0.5,
        )

        xsize = int(
            2 * radiusDeg * 60 / resoArcmin
        )

        plotTitle = (
            title 
            or f"Region {region['regionId']} "
            f"({region['type']}, {region['size']} px)"
        )
        
        hp.gnomview(
            dataMap, 
            rot=(
                region["centerGalLon"], 
                region["centerGalLat"], 
                0,
            ), 
            coord="G", 
            xsize=xsize, 
            reso=resoArcmin, 
            title=plotTitle, 
            unit=unit, 
            cmap=cmap, 
        )

        hp.graticule()

    def extract_region_cutout_data(
        self, 
        dataDict, 
        region, 
        paddingFactor=2.0, 
    ):
        """
        Return pixels and data values contained in a region cutout. 
        """

        centerVec = hp.ang2vec(
            region["centerGalLon"], 
            region["centerGalLat"], 
            lonlat=True,
        )

        pixVecs = np.array(
            hp.pix2vec(
                self.nside, 
                region["pixIds"],
            )
        ).T

        angDist = np.arccos(
            np.clip(
                pixVecs @ centerVec, 
                -1,
                1,
            )
        )

        radiusRad = angDist.max() * paddingFactor 

        cutoutPixIds = hp.query_disc(
            self.nside, 
            centerVec, 
            radiusRad, 
            inclusive=True,
        )

        inRegionMask = np.isin(
            cutoutPixIds, 
            region["pixIds"], 
        )

        cutoutData = {
            name: arr[cutoutPixIds]
            for name, arr in dataDict.items()
        }

        return {
            "pixIds": cutoutPixIds, 
            "inRegionMask": inRegionMask, 
            "data": cutoutData, 
            "radiusDeg": np.degrees(radiusRad),
        }

    def plot_footprint_vs_region(self, region, title=None):
        """
        Overlay the analysis footprint against a region's cutout, to check
        whether the region's boundary is tracing the mask edge rather than
        genuine bivariate structure.
        """
        footprintMap = self.keepMask.astype(float)
        plotTitle = title or f"Footprint mask vs Region {region['regionId']} boundary"
        self.plot_region_cutout(footprintMap, region, title=plotTitle)

    def check_footprint_edge_overlap(self, region):
        """
        Reports what fraction of a region's boundary pixels have at least
        one masked-out neighbor — i.e., sit directly against the footprint
        edge, versus being separated from it by valid, non-significant pixels.
        """
        npix = hp.nside2npix(self.nside)
    
        regionMask = np.zeros(npix, dtype=bool)
        regionMask[region["pixIds"]] = True
    
        neighbors = hp.get_all_neighbours(self.nside, region["pixIds"]).T  # (nPix, 8)
    
        boundaryPix = []
        onFootprintEdge = []
    
        for pixId, nbrs in zip(region["pixIds"], neighbors):
            nbrs = nbrs[nbrs != -1]
            outsideRegion = nbrs[~regionMask[nbrs]]  # neighbors not part of this region
    
            if len(outsideRegion) > 0:
                boundaryPix.append(pixId)
                if np.any(~self.keepMask[outsideRegion]):  # any of those neighbors masked out?
                    onFootprintEdge.append(pixId)
    
        boundaryPix = np.array(boundaryPix)
        onFootprintEdge = np.array(onFootprintEdge)
        fraction = len(onFootprintEdge) / len(boundaryPix) if len(boundaryPix) > 0 else 0.0
    
        print(f"{len(onFootprintEdge)}/{len(boundaryPix)} boundary pixels have a masked-out neighbor "
              f"({fraction:.1%})")
    
        return {"boundaryPix": boundaryPix, "onFootprintEdge": onFootprintEdge, "fraction": fraction}

    def _get_gal_lat(self):
        """
        Lazily computes and caches galactic latitude per pixel for self.nside.
        """
        if not hasattr(self, "_galLat") or self._galLat is None:
            npix = hp.nside2npix(self.nside)
            theta, _ = hp.pix2ang(self.nside, np.arange(npix))
            self._galLat = 90.0 - np.degrees(theta)
        return self._galLat
    
    def detrend_by_latitude(self, dataArray, nBins=18):
        """
        Removes a latitude trend from a full-sky array using median-binned
        galactic latitude shells (matching the 10-deg shell stratification
        used elsewhere in this project). Only pixels in self.keepMask
        contribute to the trend or receive a residual.
    
        Returns
        -------
        residual : np.ndarray
            Full-sky array, hp.UNSEEN outside self.keepMask.
        trend : np.ndarray
            Full-sky array of the fitted per-shell median, hp.UNSEEN outside self.keepMask.
        """
        if self.keepIds is None:
            raise RuntimeError(
                "Spatial weights have not been built."
                "Call build_healpix_weights() first."
            )
    
        galLat = self._get_gal_lat()
        npix = hp.nside2npix(self.nside)
    
        workingMap = dataArray.copy().astype(np.float64)
        validMask = np.zeros(npix, dtype=bool)
        validMask[self.keepIds] = np.isfinite(workingMap[self.keepIds]) & (workingMap[self.keepIds] != hp.UNSEEN)
    
        trend = np.full(npix, hp.UNSEEN)
        residual = np.full(npix, hp.UNSEEN)
    
        binEdges = np.linspace(-90, 90, nBins + 1)
    
        for i in range(nBins):
            inBin = np.where(
                validMask
                & (galLat >= binEdges[i])
                & (galLat < binEdges[i + 1])
            )[0]
            if len(inBin) > 0:
                binMedian = np.median(workingMap[inBin])
                trend[inBin] = binMedian
                residual[inBin] = workingMap[inBin] - binMedian
    
        return residual, trend
    
    def generate_latitude_aware_surrogates(self, dataMap, nSurrogates=100, lmax=None, nBins=18):
        """
        Generates Gaussian surrogates that preserve the input map's
        latitude trend exactly (real trend, not resimulated) while
        randomizing the residual structure via power-spectrum-matched
        synfast. Use in place of generate_gaussian_surrogates() when
        testing for structure beyond a known latitude confound.
        """
        if lmax is None:
            lmax = 3 * self.nside - 1
    
        residual, trend = self.detrend_by_latitude(dataMap, nBins=nBins)
    
        npix = hp.nside2npix(self.nside)
        validMask = residual != hp.UNSEEN
    
        workingResidual = residual.copy()
        fillValue = np.mean(workingResidual[validMask])
        workingResidual[~validMask] = fillValue
    
        cl = hp.anafast(workingResidual, lmax=lmax)
    
        surrogates = []
        for i in range(nSurrogates):
            surrogateResidual = hp.synfast(cl, nside=self.nside, lmax=lmax)
    
            surrogate = np.full(npix, hp.UNSEEN)
            surrogate[validMask] = surrogateResidual[validMask] + trend[validMask]
    
            surrogates.append(surrogate)
    
        return surrogates