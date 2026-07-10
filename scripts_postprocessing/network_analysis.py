import os
import json
import dynetan as dna
from dynetan.toolkit import getSelFromNode
import networkx as nx
from itertools import islice
from operator import itemgetter
import argparse

def create_ag_parser():
        parser = argparse.ArgumentParser(description="Program to calculate correlation matrix from MD simulations")
        parser.add_argument("-p", "--pdb", type=str, help="the path of pdb or topology file of the md simulation")
        parser.add_argument("-t", "--trajectory", type=str,help="the path of the trajectory file")
        parser.add_argument("-ids", "--seg_ids", nargs="+", default=["A"], help="The segments to analyze or make graphs")
        parser.add_argument("-jn", "--job_name", default="protein", help="The name of the job for outputs")
        parser.add_argument("-nw", "--number_windows",type=int, default=4, help="The number of windows for the calculation")
        parser.add_argument("-s", "--sampled_frames",type=int, default=10, help="The number of frames to sample per window")
        parser.add_argument("-c", "--cutoff_distance",type=float, default=4.5, help="The distance in angstroms between the nodes")
        parser.add_argument("-n", "--n_jobs", type=int, default=8, help="The number of threads to use for the calculation")
        parser.add_argument("-nd","--new_dcd", action='store_true',help='Wheter to store a new dcd with stride')
        parser.add_argument("-d", "--dcd_stride", type=int, default=1, help="The stride for dcd trajectory")
        parser.add_argument("-o", "--out", required=True, type=str, help="The path of where to leave the output files")
        parser.add_argument("--json", type=str, default=None, help="Path to write JSON result file")

        return parser

def make_network_analysis(pdb_path, trajectory_path, seg_ids, job_name, n_windows, sampled_frames, cutoff, n_cores, new_dcd, dcd_stride, out, json_output=None):
        pdb = pdb_path
        dcd = trajectory_path
        output_path = out
        filenameroot = job_name
        final_output = os.path.join(output_path,job_name)
        contactPersistence = 0.75
        usrNodeGroups = {}
        dnap = 	dna.proctraj.DNAproc(notebookMode=False)
        dnap.setNumWinds(n_windows)
        dnap.setNumSampledFrames(sampled_frames)
        dnap.setCutoffDist(cutoff)
        dnap.setContactPersistence(contactPersistence)
        dnap.setSolvNames(["HOH", "WAT", "SOL"])
        dnap.setSegIDs(seg_ids)
        dnap.setNodeGroups(usrNodeGroups)
        print(f"Loading topology file {pdb} and trajectory file(s) {dcd}.")
        dnap.loadSystem(pdb, dcd)
        print("System loaded.")
        print("MDAnalysis universe:", dnap.getU().trajectory)
        dnap.checkSystem()
        dnap.selectSystem(withSolvent=False)
        dnap.prepareNetwork()

        print("Aligning trajectory...")
        dnap.alignTraj()

        print("Finding contacts...")
        dnap.findContacts(stride=1, verbose=True)
        dnap.checkContactMat(verbose=1)
        
        dnap.calcCor(ncores=n_cores, verbose=True)
        dnap.calcCartesian(backend="serial")
        dnap.calcGraphInfo()
        print("Graph with {} nodes and {} edges".format(len(dnap.nxGraphs[0].nodes),
                                                len(dnap.nxGraphs[0].edges)))

        for win in range(dnap.numWinds):
            print("----- Window {} -----".format(win))
            print("Density:", round( nx.density(dnap.nxGraphs[win]), 4) )
            print("Transitivity:", round( nx.transitivity(dnap.nxGraphs[win]), 4) )
            print()

        for win in range(dnap.numWinds):
            print("----- Window {} -----".format(win))

            sorted_degree = sorted(dnap.getDegreeDict(win).items(), key=itemgetter(1), reverse=True)

            print("Top 5 nodes by degree: [node --> degree : selection]")
            for n,d in sorted_degree[:5]:
                print("{0:>4} --> {1:>2} : {2}".format(n, d, getSelFromNode(n, dnap.nodesAtmSel)))

            print()

        print("Calculating optimal paths...")
        dnap.calcOptPaths(ncores=n_cores)

        print("Calculating edge betweenness...")
        dnap.calcBetween(ncores=n_cores)

        print("Here are the top 5 pairs of nodes based on Betweenness values, ""compared to their correlation values (in Window 0):")

        for k, v in islice(dnap.btws[0].items(),5):
            node_pair = k
            btw = round(v, 3)
            corr = round(dnap.corrMatAll[0, k[0], k[1]], 3)
            print(f"\tNodes {node_pair} have betweenness {btw} and correlation {corr}.")

        dnap.calcEigenCentral()
        dnap.calcCommunities()

        print("Here are the top 5 communities based on number of nodes:")
        # Sort communities based on number of nodes
        for comIndx in islice(dnap.nodesComm[0]["commOrderSize"], 5):
            print("Modularity Class {0:>2}: {1:>3} nodes.".format(comIndx, len(dnap.nodesComm[0]["commNodes"][comIndx])))

        print("Here are the top 5 communities based on Eigenvector Centrality:")
        for comIndx in islice(dnap.nodesComm[0]["commOrderEigenCentr"], 5):
            print("Modularity Class {0} ({1} nodes) Sorted by Eigenvector Centrality:".format(
                                                                    comIndx,
                                                                len(dnap.nodesComm[0]["commNodes"][comIndx])))
            for node in dnap.nodesComm[0]["commNodes"][comIndx][:5]:
                print("Name: {0:>4} | Degree: {1:>2} | Eigenvector Centrality: {2}".format(
                   node, dnap.nxGraphs[win].nodes[node]['degree'], dnap.nxGraphs[win].nodes[node]['eigenvector']))
            print()

        dnap.saveData(final_output)

        if new_dcd:

            num_atoms = dnap.workU.atoms.n_atoms
            num_frames = len(dnap.workU.trajectory[::dcd_stride])

            print(f"We will save {num_atoms} heavy atoms and {num_frames} frames.")

            dnap.saveReducedTraj(final_output, stride=dcd_stride)

        n_nodes = len(dnap.nxGraphs[0].nodes)
        n_edges = len(dnap.nxGraphs[0].edges)

        result = {
            "step": "network_analysis",
            "status": "completed",
            "output_prefix": final_output,
            "n_windows": n_windows,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
        }

        if json_output:
            with open(json_output, "w") as f:
                json.dump(result, f, indent=2)

        return result


if __name__ == "__main__":
        parser = create_ag_parser()
        args = parser.parse_args()
        make_network_analysis(args.pdb, args.trajectory, args.seg_ids, args.job_name, args.number_windows, args.sampled_frames, args.cutoff_distance, args.n_jobs, args.new_dcd, args.dcd_stride, args.out, args.json)


