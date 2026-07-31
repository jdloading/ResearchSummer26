import numpy as np 
import healpy as hp 
import libpysal 

from scipy import sparse 
from scipy.sparse.csgraph import connected_components
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

    def lisa_bv_report(
        self, 
        xArray,
        yArray, 
        label, 
        permutations=None
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

        lisaBV = Moran_Local_BV(
            xValid,
            yValid, 
            self.w, 
            permutations=permutations,
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
            quadCodes = sigQuadCodes[mask]

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
    def build_lisa_bv_map(
        self, 
        lisaBV, 
        alpha=None,
    ):
        """
        Map LISA-BV results onto a full-sky HEALPIX array.
        
        Codes: 
        0 = not sig
        1 = HH 
        2 = LH 
        3 = LL 
        4 = HL
        
        Pixels outside the retained analysis region are set to hp.UNSEEN.
        """
        if self.keepIds is None: 
            raise RuntimeError(
                "Spatial weights have not been built."
                "Call build_healpix_weights() first. "
            )
        if alpha is None: 
            alpha = self.alpha 

        npix = hp.nside2npix(self.nside) 

        quadMap = np.full(
            npix, hp.UNSEEN,
        )

        sig = lisaBV.p_sim < alpha 
        code = np.where(
            sig,
            lisaBV.q, 
            0,
        )

        quadMap[self.keepIds]=code 

        return quadMap 

    def plot_region(
        self, 
        regionId, 
        regionPixIds, 
        regionLabels,
    ):
        """
        Plot a signle connnected region on a full-sky HEALPix map. 
        """

        mask = regionLabels == regionId 
        pixIds = regionPixIds[mask]

        npix = hp.nside2npix(self.nside) 

        highlightMap = np.full(
            npix, hp.UNSEEN, 
        )

        highlightMap[pixIds] = 1

        hp.newvisufunc.projview(
            highlightMap, 
            coord=["G"], 
            graticule=True, 
            graticule_labels=True, 
            title=f"Region {regionId} ({len(pixIds)} pixels)",
            cbar=False,
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