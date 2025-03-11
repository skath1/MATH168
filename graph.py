import chess.pgn
import networkx as nx
import matplotlib.pyplot as plt
from analyze import pgn_file

def build_move_network(pgn_file, max_games=20):
    G = nx.DiGraph()  # Directed graph for move transitions

    with open(pgn_file, encoding="utf-8") as file:
        game_count = 0

        while game_count < max_games:
            game = chess.pgn.read_game(file)
            if game is None:
                break

            game_count += 1
            if game_count % 500 == 0:
                print(f"Processed {game_count} games")

            # Extract move sequence
            node = game
            previous_move = None
            board = node.board()

            for move in node.mainline_moves():
                san_move = board.san(move)
                board.push(move)

                # Create transitions between consecutive moves
                if previous_move:
                    if G.has_edge(previous_move, san_move):
                        G[previous_move][san_move]['weight'] += 1
                    else:
                        G.add_edge(previous_move, san_move, weight=1)

                previous_move = san_move  # Update previous move

    return G

def build_position_network(pgn_file, max_games=20):
    """Build a network where nodes are board positions and edges are moves between them"""
    G = nx.DiGraph()  # Directed graph for position transitions
    
    with open(pgn_file, encoding="utf-8") as file:
        game_count = 0
        
        while game_count < max_games:
            game = chess.pgn.read_game(file)
            if game is None:
                break
                
            game_count += 1
            if game_count % 500 == 0:
                print(f"Processed {game_count} games")
                
            # Track game outcome for later analysis
            result = game.headers.get("Result", "*")
            
            # Extract position sequence
            board = game.board()
            previous_position = board.fen().split(' ')[0]  # Just the position part of FEN
            
            for move in game.mainline_moves():
                san_move = board.san(move)
                board.push(move)
                current_position = board.fen().split(' ')[0]
                
                # Add nodes if they don't exist
                if not G.has_node(previous_position):
                    G.add_node(previous_position, visits=1)
                else:
                    G.nodes[previous_position]['visits'] += 1
                    
                if not G.has_node(current_position):
                    G.add_node(current_position, visits=1)
                else:
                    G.nodes[current_position]['visits'] += 1
                
                # Create transitions between positions
                if G.has_edge(previous_position, current_position):
                    G[previous_position][current_position]['weight'] += 1
                else:
                    G.add_edge(previous_position, current_position, 
                              weight=1, 
                              move=san_move)
                
                previous_position = current_position
                
    return G

def analyze_move_network(G):
    print("\n--- Move Network Analysis ---")
    
    # 1. Centrality Measures
    print("Calculating centrality measures...")
    # Adjust k to be the minimum of 500 or the number of nodes
    k = min(500, len(G.nodes()))
    betweenness = nx.betweenness_centrality(G, weight='weight', k=k)
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())

    # Find most influential move
    most_influential = max(betweenness, key=betweenness.get)
    print(f"Most Influential Move (Betweenness Centrality): {most_influential} (Score: {betweenness[most_influential]:.4f})")
    
    # Find most popular move
    most_popular = max(in_degree, key=in_degree.get)
    print(f"Most Popular Move (In-Degree Centrality): {most_popular} (In-Degree: {in_degree[most_popular]})")
    
    # Find move leading to the most variations
    most_variations = max(out_degree, key=out_degree.get)
    print(f"Move Leading to Most Variations (Out-Degree Centrality): {most_variations} (Out-Degree: {out_degree[most_variations]})")

    # 2. Clustering Coefficient
    print("Calculating clustering coefficient...")
    try:
        # Convert to undirected for clustering coefficient
        G_undirected = G.to_undirected()
        clustering = nx.clustering(G_undirected)
        avg_clustering = nx.average_clustering(G_undirected)
        print(f"Average Clustering Coefficient: {avg_clustering:.4f}")
        
        # Find moves with highest clustering (most interconnected)
        most_clustered = max(clustering, key=clustering.get)
        print(f"Most Interconnected Move: {most_clustered} (Clustering: {clustering[most_clustered]:.4f})")
    except Exception as e:
        print(f"Clustering calculation error: {e}")
    
    # 3. Shortest Path Analysis
    print("Analyzing shortest paths...")
    try:
        # Sample some nodes for path analysis (full analysis would be too computationally expensive)
        sample_nodes = list(G.nodes())[:min(100, len(G.nodes()))]
        avg_path_length = nx.average_shortest_path_length(G, weight='weight')
        print(f"Average Shortest Path Length: {avg_path_length:.4f}")
    except Exception as e:
        print(f"Path analysis error: {e}")
    
    # 4. Community Detection
    print("Detecting communities...")
    try:
        # Convert to undirected for community detection
        G_undirected = G.to_undirected()
        communities = nx.community.greedy_modularity_communities(G_undirected)
        print(f"Number of move communities detected: {len(communities)}")
        
        # Print largest communities (opening/strategy clusters)
        for i, community in enumerate(list(communities)[:3]):
            print(f"Community {i+1} size: {len(community)} moves")
            print(f"Sample moves: {list(community)[:5]}")
    except Exception as e:
        print(f"Community detection error: {e}")

    return betweenness, in_degree, out_degree

def visualize_move_network(G, betweenness, in_degree, out_degree):
    print("Generating network visualization...")
    
    # Extract node sizes based on in-degree centrality
    node_sizes = [in_degree.get(node, 0) * 10 for node in G]  # Scale size for visibility

    # Limit visualization to most significant nodes for clarity
    if len(G) > 200:
        # Create a subgraph of the most important nodes
        significant_nodes = sorted(in_degree, key=in_degree.get, reverse=True)[:200]
        G_viz = G.subgraph(significant_nodes)
        node_sizes = [in_degree.get(node, 0) * 10 for node in G_viz]
    else:
        G_viz = G

    # Draw the graph
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G_viz, seed=42)
    
    # Draw nodes with size based on in-degree
    nx.draw(G_viz, pos, with_labels=True, node_size=node_sizes, 
            edge_color='gray', font_size=8, arrowsize=10, alpha=0.7)

    # Highlight key moves
    key_nodes = []
    if betweenness:
        most_influential = max(betweenness, key=betweenness.get)
        if most_influential in G_viz:
            key_nodes.append(('Most Influential', most_influential, 'red'))
    
    if in_degree:
        most_popular = max(in_degree, key=in_degree.get)
        if most_popular in G_viz:
            key_nodes.append(('Most Popular', most_popular, 'blue'))
    
    if out_degree:
        most_variations = max(out_degree, key=out_degree.get)
        if most_variations in G_viz:
            key_nodes.append(('Most Variations', most_variations, 'green'))
    
    # Draw highlighted nodes
    for label, node, color in key_nodes:
        nx.draw_networkx_nodes(G_viz, pos, nodelist=[node], node_color=color, 
                              node_size=in_degree.get(node, 5) * 15)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=10, label=label)
                      for label, _, color in key_nodes]
    plt.legend(handles=legend_elements, loc="upper right")
    
    plt.title("Chess Move Network")
    plt.savefig("chess_move_network.png", dpi=300)
    plt.show()

def visualize_position_network(G, top_n=100):
    """Visualize the top N most visited positions in the network"""
    print("Generating position network visualization...")
    
    # Extract the most common positions
    positions_by_visits = sorted(G.nodes(data=True), 
                               key=lambda x: x[1].get('visits', 0), 
                               reverse=True)[:top_n]
    
    # Create a subgraph of just these positions
    sub_G = G.subgraph([p[0] for p in positions_by_visits])
    
    # Draw the graph
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(sub_G, seed=42)
    
    # Node sizes based on visit count
    node_sizes = [sub_G.nodes[node].get('visits', 1)/5 for node in sub_G]
    
    # Edge widths based on weight
    edge_widths = [sub_G[u][v].get('weight', 1)/10 for u, v in sub_G.edges()]
    
    nx.draw(sub_G, pos, with_labels=False, node_size=node_sizes, 
           width=edge_widths, edge_color='gray', alpha=0.7)
    
    # Add labels for the top 10 positions
    top_positions = [p[0] for p in positions_by_visits[:10]]
    labels = {pos: f"Pos {i+1}" for i, pos in enumerate(top_positions)}
    nx.draw_networkx_labels(sub_G, pos, labels=labels, font_size=12)
    
    plt.title("Chess Position Network (Top Positions)")
    plt.savefig("chess_position_network.png", dpi=300)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Build the move-based network
    print("Building move network...")
    G_moves = build_move_network(pgn_file, max_games=20)
    print("Move Network construction complete.")

    # Analyze the move-based network
    betweenness, in_degree, out_degree = analyze_move_network(G_moves)

    # Visualize the move-based network
    visualize_move_network(G_moves, betweenness, in_degree, out_degree)
    
    # Build the position-based network
    print("\nBuilding position network...")
    G_positions = build_position_network(pgn_file, max_games=20)
    print("Position Network construction complete.")
    
    # Visualize the position network
    visualize_position_network(G_positions, top_n=100)
    
    # Save networks for future analysis
    print("Saving networks to files...")
    nx.write_gexf(G_moves, "chess_move_network.gexf")
    nx.write_gexf(G_positions, "chess_position_network.gexf")
    print("Analysis complete!")
