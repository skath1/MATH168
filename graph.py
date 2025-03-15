import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import chess.pgn
import io
import seaborn as sns
from collections import defaultdict
from community import community_louvain


class ChessNetwork:
    def __init__(self):
        """Initialize the chess network graph."""
        self.G = nx.DiGraph()
        self.positions_map = {}  # Maps FEN positions to node IDs
        self.node_counter = 0
        self.games_analyzed = 0

    def add_game(self, pgn_text):
        """
        Parse a PGN game and add its moves to the network.
        
        Args:
            pgn_text (str): Chess game in PGN format
        """
        pgn = io.StringIO(pgn_text)
        game = chess2.pgn.read_game(pgn)
        
        if game is None:
            return False
        
        self.games_analyzed += 1
        
        # Track the board position
        board = game.board()
        current_position = board.fen().split(' ')[0]  # Only use piece positions part of FEN
        
        # Add initial position if not already in the graph
        if current_position not in self.positions_map:
            self.positions_map[current_position] = self.node_counter
            self.G.add_node(self.node_counter, fen=current_position, visits=1)
            self.node_counter += 1
        else:
            # Increment visits for this position
            node_id = self.positions_map[current_position]
            self.G.nodes[node_id]['visits'] += 1
        
        # Process each move in the game
        for move in game.mainline_moves():
            source_node = self.positions_map[current_position]
            
            # Make the move on the board
            board.push(move)
            new_position = board.fen().split(' ')[0]
            
            # Add new position if not already in the graph
            if new_position not in self.positions_map:
                self.positions_map[new_position] = self.node_counter
                self.G.add_node(self.node_counter, fen=new_position, visits=1)
                self.node_counter += 1
            else:
                # Increment visits for this position
                node_id = self.positions_map[new_position]
                self.G.nodes[node_id]['visits'] += 1
            
            target_node = self.positions_map[new_position]
            
            # Add or update the edge
            if self.G.has_edge(source_node, target_node):
                self.G[source_node][target_node]['weight'] += 1
            else:
                self.G.add_edge(source_node, target_node, weight=1, move=move.uci())
            
            current_position = new_position
        
        return True

    def load_pgn_file(self, file_path, max_games=None):
        """
        Load games from a PGN file and add them to the network.
        
        Args:
            file_path (str): Path to the PGN file
            max_games (int, optional): Maximum number of games to load
        """
        games_loaded = 0
        
        with open(file_path, 'r') as f:
            pgn_text = ""
            for line in f:
                pgn_text += line
                
                # Empty line might indicate end of a game
                if line.strip() == "" and pgn_text.strip() != "":
                    if "1." in pgn_text:  # Simple check to see if there are moves
                        if self.add_game(pgn_text):
                            games_loaded += 1
                            if max_games and games_loaded >= max_games:
                                break
                    pgn_text = ""
        
        print(f"Loaded {games_loaded} games into the network")
        print(f"Network has {len(self.G.nodes)} positions and {len(self.G.edges)} transitions")
        
    def calculate_metrics(self):
        """Calculate various network metrics and add them as node attributes."""
        print("Calculating network metrics...")
        
        # Degree Centrality
        in_degree = dict(self.G.in_degree(weight='weight'))
        out_degree = dict(self.G.out_degree(weight='weight'))
        
        # Add degree metrics to nodes
        for node in self.G.nodes():
            self.G.nodes[node]['in_degree'] = in_degree.get(node, 0)
            self.G.nodes[node]['out_degree'] = out_degree.get(node, 0)
            self.G.nodes[node]['total_degree'] = in_degree.get(node, 0) + out_degree.get(node, 0)
        
        # Betweenness Centrality (can be computationally expensive for large networks)
        if len(self.G) < 10000:  # Only calculate for smaller networks
            print("Calculating betweenness centrality...")
            betweenness = nx.betweenness_centrality(self.G, weight='weight', k=min(100, len(self.G)))
            nx.set_node_attributes(self.G, betweenness, 'betweenness')
        
        # Clustering Coefficient (for undirected version of the graph)
        print("Calculating clustering coefficients...")
        G_undirected = self.G.to_undirected()
        clustering = nx.clustering(G_undirected, weight='weight')
        nx.set_node_attributes(self.G, clustering, 'clustering')
        
        # Community Detection
        print("Detecting communities...")
        communities = community_louvain.best_partition(G_undirected)
        nx.set_node_attributes(self.G, communities, 'community')
        
        print("Metrics calculation complete")

    def get_top_positions(self, metric='visits', n=10):
        """
        Get the top positions based on a specified metric.
        
        Args:
            metric (str): The metric to sort by ('visits', 'in_degree', 'out_degree', 'betweenness', etc.)
            n (int): Number of positions to return
            
        Returns:
            DataFrame: Top positions with their metrics
        """
        nodes_data = []
        for node, data in self.G.nodes(data=True):
            if metric in data:
                nodes_data.append({
                    'node_id': node,
                    'fen': data.get('fen', ''),
                    metric: data.get(metric, 0),
                    'visits': data.get('visits', 0),
                    'in_degree': data.get('in_degree', 0),
                    'out_degree': data.get('out_degree', 0),
                    'betweenness': data.get('betweenness', 0),
                    'clustering': data.get('clustering', 0),
                    'community': data.get('community', -1)
                })
        
        df = pd.DataFrame(nodes_data)
        return df.sort_values(by=metric, ascending=False).head(n)

    def get_common_transitions(self, n=10):
        """
        Get the most common position transitions.
        
        Args:
            n (int): Number of transitions to return
            
        Returns:
            DataFrame: Top transitions with their weights
        """
        edges_data = []
        for u, v, data in self.G.edges(data=True):
            edges_data.append({
                'source': u,
                'target': v,
                'weight': data.get('weight', 0),
                'move': data.get('move', ''),
                'source_fen': self.G.nodes[u].get('fen', ''),
                'target_fen': self.G.nodes[v].get('fen', '')
            })
        
        df = pd.DataFrame(edges_data)
        return df.sort_values(by='weight', ascending=False).head(n)

    def find_shortest_paths(self, start_fen, end_fen, k=3):
        """
        Find the k shortest paths between two positions.
        
        Args:
            start_fen (str): Starting position in FEN format
            end_fen (str): Ending position in FEN format
            k (int): Number of paths to find
            
        Returns:
            list: List of paths (each path is a list of node IDs)
        """
        if start_fen not in self.positions_map or end_fen not in self.positions_map:
            return []
        
        start_node = self.positions_map[start_fen]
        end_node = self.positions_map[end_fen]
        
        try:
            paths = list(nx.shortest_simple_paths(self.G, start_node, end_node, weight='weight'))
            return paths[:k]
        except nx.NetworkXNoPath:
            return []

    def visualize_network(self, max_nodes=100, metric='visits', min_edge_weight=2):
        """
        Visualize the chess network.
        
        Args:
            max_nodes (int): Maximum number of nodes to display
            metric (str): Metric to determine node size
            min_edge_weight (int): Minimum edge weight to display
        """
        # Create a subgraph with the top nodes by the specified metric
        top_nodes = self.get_top_positions(metric=metric, n=max_nodes)['node_id'].tolist()
        subgraph = self.G.subgraph(top_nodes)
        
        # Filter edges by minimum weight
        edges_to_keep = [(u, v) for u, v, d in subgraph.edges(data=True) if d.get('weight', 0) >= min_edge_weight]
        filtered_graph = subgraph.edge_subgraph(edges_to_keep)
        
        # Set up the plot
        plt.figure(figsize=(12, 10))
        
        # Node sizes based on the metric
        node_sizes = [filtered_graph.nodes[n].get(metric, 1) * 10 for n in filtered_graph.nodes()]
        
        # Node colors based on community
        node_colors = [filtered_graph.nodes[n].get('community', 0) for n in filtered_graph.nodes()]
        
        # Edge weights for thickness
        edge_weights = [filtered_graph[u][v].get('weight', 1) / 2 for u, v in filtered_graph.edges()]
        
        # Create the layout
        pos = nx.spring_layout(filtered_graph, seed=42)
        
        # Draw the network
        nx.draw_networkx_nodes(filtered_graph, pos, node_size=node_sizes, node_color=node_colors, 
                              cmap=plt.cm.viridis, alpha=0.8)
        nx.draw_networkx_edges(filtered_graph, pos, width=edge_weights, alpha=0.5, 
                              edge_color='gray', arrows=True, arrowsize=10)
        
        plt.title(f"Chess Position Network (showing top {len(filtered_graph)} positions by {metric})")
        plt.axis('off')
        plt.tight_layout()
        
        return plt

    def plot_degree_distribution(self):
        """Plot the degree distribution of the network."""
        degrees = [d for n, d in self.G.degree()]
        
        plt.figure(figsize=(10, 6))
        plt.hist(degrees, bins=30, alpha=0.7)
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.title('Degree Distribution')
        plt.grid(True, alpha=0.3)
        
        return plt

    def plot_community_sizes(self):
        """Plot the distribution of community sizes."""
        communities = nx.get_node_attributes(self.G, 'community')
        if not communities:
            print("No community data available. Run calculate_metrics() first.")
            return None
        
        community_sizes = defaultdict(int)
        for community_id in communities.values():
            community_sizes[community_id] += 1
        
        # Convert to DataFrame for plotting
        df = pd.DataFrame({
            'Community': list(community_sizes.keys()),
            'Size': list(community_sizes.values())
        }).sort_values('Size', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Community', y='Size', data=df.head(20))
        plt.title('Top 20 Community Sizes')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt

    def plot_heatmap(self, metric='visits'):
        """
        Create a heatmap visualization of the chess board with positions colored by a metric.
        
        Args:
            metric (str): Metric to use for coloring ('visits', 'in_degree', etc.)
        """
        # Get top positions
        top_positions = self.get_top_positions(metric=metric, n=100)
        
        # Create an 8x8 grid for the chess board
        heatmap_data = np.zeros((8, 8))
        
        for _, row in top_positions.iterrows():
            fen = row['fen']
            value = row[metric]
            
            # Parse FEN to get piece positions
            ranks = fen.split('/')
            for rank_idx, rank in enumerate(ranks):
                file_idx = 0
                for char in rank:
                    if char.isdigit():
                        file_idx += int(char)
                    else:
                        # Add the value to the corresponding square
                        if file_idx < 8 and rank_idx < 8:  # Ensure within bounds
                            heatmap_data[rank_idx][file_idx] += value
                        file_idx += 1
        
        # Plot the heatmap
        plt.figure(figsize=(10, 10))
        sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False)
        plt.title(f'Chess Board Heatmap by {metric}')
        plt.xlabel('File (a-h)')
        plt.ylabel('Rank (1-8)')
        plt.xticks(np.arange(8) + 0.5, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        plt.yticks(np.arange(8) + 0.5, ['8', '7', '6', '5', '4', '3', '2', '1'])
        
        return plt


# Example usage
if __name__ == "__main__":
    # Create a chess network
    chess_net = ChessNetwork()
    
    # Load games from your existing PGN file
    print("Loading games from PGN file...")
    chess_net.load_pgn_file(pgn_file, max_games=50)  # Using 50 games for faster analysis
    
    # Calculate network metrics
    chess_net.calculate_metrics()
    
    # Get top positions by visits
    top_positions = chess_net.get_top_positions(metric='visits', n=10)
    print("\nTop positions by visits:")
    print(top_positions[['fen', 'visits', 'in_degree', 'out_degree']])
    
    # Get most common transitions
    common_transitions = chess_net.get_common_transitions(n=10)
    print("\nMost common transitions:")
    print(common_transitions[['move', 'weight']])
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Visualize the network
    plt_network = chess_net.visualize_network(max_nodes=50)
    plt_network.savefig("chess_network.png")
    
    # Plot degree distribution
    plt_degrees = chess_net.plot_degree_distribution()
    plt_degrees.savefig("degree_distribution.png")
    
    # Plot community sizes
    plt_communities = chess_net.plot_community_sizes()
    if plt_communities:
        plt_communities.savefig("community_sizes.png")
    
    # Plot heatmap
    plt_heatmap = chess_net.plot_heatmap(metric='visits')
    plt_heatmap.savefig("chess_heatmap.png")
    
    print("\nAnalysis complete. Visualizations saved as PNG files.")
