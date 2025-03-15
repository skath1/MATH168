import chess.pgn
import chess
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from analyze import pgn_file
import random

# Function to parse PGN file and extract move sequences
def parse_pgn(file_path, max_games=1000):  # Set default to 1000 games
    games = []
    print(f"Loading first {max_games} games from PGN file...")
    with open(file_path) as pgn_file:
        for _ in tqdm(range(max_games)):
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    print(f"Loaded {len(games)} games")
    return games

# Function to create a move transition network
def create_move_network(games, min_position_count=5):
    G = nx.DiGraph()
    position_count = defaultdict(int)
    position_wins = defaultdict(int)  # Track wins for each position
    position_games = defaultdict(int)  # Track total games for each position

    for game in games:
        board = game.board()
        previous_position = None
        
        # Get game result
        result = game.headers.get("Result", "*")
        is_white_win = result == "1-0"
        is_black_win = result == "0-1"
        
        # Skip games with no clear winner
        if not (is_white_win or is_black_win):
            continue

        positions_this_game = set()  # Track positions in this game
        
        for move in game.mainline_moves():
            board.push(move)
            current_position = board.fen()
            positions_this_game.add(current_position)
            
            # Update position count
            position_count[current_position] += 1

            # Add edge between previous and current position
            if previous_position is not None:
                if G.has_edge(previous_position, current_position):
                    G[previous_position][current_position]['weight'] += 1
                else:
                    G.add_edge(previous_position, current_position, weight=1)

            previous_position = current_position
        
        # Update win counts for all positions in this game
        for pos in positions_this_game:
            position_games[pos] += 1
            if (is_white_win and board.turn) or (is_black_win and not board.turn):
                position_wins[pos] += 1

    # Filter out positions and add win rates
    filtered_G = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if position_count[u] >= min_position_count and position_count[v] >= min_position_count:
            # Calculate win rates
            u_win_rate = position_wins[u] / position_games[u] if position_games[u] > 0 else 0
            v_win_rate = position_wins[v] / position_games[v] if position_games[v] > 0 else 0
            
            # Add edge with win rate data
            filtered_G.add_edge(u, v, 
                              weight=data['weight'],
                              source_win_rate=u_win_rate,
                              target_win_rate=v_win_rate)
            
            # Add win rate data to nodes
            filtered_G.nodes[u]['win_rate'] = u_win_rate
            filtered_G.nodes[u]['games'] = position_games[u]
            filtered_G.nodes[v]['win_rate'] = v_win_rate
            filtered_G.nodes[v]['games'] = position_games[v]

    return filtered_G, position_count

# Function to analyze the network
def analyze_network(G):
    print("\n=== Advanced Network Analysis ===")
    
    # 1. Basic Network Metrics
    print("\n1. Network Structure")
    print("-------------------")
    print(f"Total positions analyzed: {G.number_of_nodes()}")
    print(f"Total moves (transitions): {G.number_of_edges()}")
    print(f"Network density: {nx.density(G):.3f}")
    print(f"Average degree: {sum(dict(G.degree()).values())/G.number_of_nodes():.2f}")
    
    # 2. Centrality Measures
    print("\n2. Position Importance Metrics")
    print("--------------------------")
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G)
    
    # Find key positions using different metrics
    key_positions = set()
    for metric, values in [
        ("Degree", degree_centrality),
        ("Betweenness", betweenness_centrality),
        ("PageRank", pagerank)
    ]:
        top_pos = max(values.items(), key=lambda x: x[1])
        key_positions.add(top_pos[0])
        print(f"\n{metric} Analysis:")
        print(f"Most important position: {get_opening_name(top_pos[0])}")
        print(f"Value: {top_pos[1]:.4f}")
    
    # 3. Community Analysis
    print("\n3. Opening Communities")
    print("-------------------")
    communities = nx.community.louvain_communities(G)
    modularity = nx.community.modularity(G, communities)
    print(f"Number of distinct opening groups: {len(communities)}")
    print(f"Community structure strength (modularity): {modularity:.3f}")
    
    return {
        'degree_centrality': degree_centrality,
        'betweenness_centrality': betweenness_centrality,
        'pagerank': pagerank,
        'communities': communities,
        'modularity': modularity,
        'key_positions': key_positions
    }

def get_opening_name(fen):
    """Convert FEN to common opening name if known"""
    # Extract just the piece positions part of the FEN (before the first space)
    position = fen.split(' ')[0]
    
    opening_dict = {
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR": "King's Pawn Opening",
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR": "Queen's Pawn Opening",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR": "Sicilian Defense",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR": "King's Pawn Defense",
        "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR": "Queen's Pawn Defense",
        # Common responses
        "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR": "Alekhine's Defense",
        "rnbqkbnr/ppp1pppp/3p4/8/4P3/8/PPPP1PPP/RNBQKBNR": "Caro-Kann Defense",
        "rnbqkbnr/pppppp1p/6p1/8/4P3/8/PPPP1PPP/RNBQKBNR": "Modern Defense",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R": "Ruy Lopez Setup",
        "rnbqkb1r/ppp1pppp/5n2/3p4/3P4/8/PPP1PPPP/RNBQKBNR": "King's Indian Defense",
        # Add more openings as needed
    }
    return opening_dict.get(position, "Unknown Position")

def visualize_network(G, analysis_results):
    # Common layout for all visualizations
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Common label settings
    labels = {node: get_opening_name(node) for node in G.nodes()}
    base_font_size = 6
    
    # 1. Degree Centrality Visualization
    plt.figure(figsize=(15, 15))
    node_size = [analysis_results['degree_centrality'][node] * 8000 for node in G.nodes]
    nx.draw(G, pos, 
            with_labels=True,
            labels=labels,
            node_size=node_size,
            font_size=base_font_size,
            node_color='#1f77b4',
            width=0.3,
            alpha=0.9,
            edge_color='#cccccc')
    plt.title("Chess Opening Network\nNode Size = Position Frequency", fontsize=16, pad=20)
    plt.savefig("network_degree.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Betweenness Centrality Visualization
    fig, ax = plt.subplots(figsize=(15, 15))
    node_color = [analysis_results['betweenness_centrality'][node] for node in G.nodes]
    nx.draw(G, pos, 
            with_labels=True,
            labels=labels,
            node_size=300,
            font_size=base_font_size,
            width=0.3,
            alpha=0.9,
            edge_color='#cccccc',
            node_color=node_color, 
            cmap=plt.cm.viridis,
            ax=ax)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    sm.set_array(node_color)
    plt.colorbar(sm, ax=ax, label="Betweenness Centrality")
    plt.title("Chess Opening Network\nNode Color = Position Importance", fontsize=16, pad=20)
    plt.savefig("network_betweenness.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Community Visualization
    plt.figure(figsize=(15, 15))
    communities = analysis_results['communities']
    community_colors = {node: i for i, comm in enumerate(communities) for node in comm}
    node_colors = [community_colors[node] for node in G.nodes]
    
    nx.draw(G, pos, 
            with_labels=True,
            labels=labels,
            node_size=300,
            font_size=base_font_size,
            width=0.3,
            alpha=0.9,
            edge_color='#cccccc',
            node_color=node_colors, 
            cmap=plt.cm.Set3)

    # Add community legend
    unique_colors = sorted(set(node_colors))
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=plt.cm.Set3(color/len(unique_colors)), 
                                 label=f'Community {color+1}', markersize=10)
                      for color in unique_colors]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Chess Opening Network\nNode Colors = Opening Communities", fontsize=16, pad=20)
    plt.savefig("network_communities.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Victory Paths Visualization
    if 'victory_paths' in analysis_results and analysis_results['victory_paths']:
        plt.figure(figsize=(15, 15))
        # Draw base network in light gray
        nx.draw(G, pos,
                with_labels=True,
                labels=labels,
                node_size=100,
                font_size=base_font_size,
                width=0.1,
                alpha=0.3,
                edge_color='lightgray',
                node_color='lightgray')
        
        # Highlight victory paths
        for path, length, win_rate in analysis_results['victory_paths'][:3]:  # Show top 3 paths
            path_edges = list(zip(path[:-1], path[1:]))
            path_color = plt.cm.RdYlGn(win_rate)
            
            nx.draw_networkx_nodes(G, pos,
                                 nodelist=path,
                                 node_color=[path_color],
                                 node_size=300,
                                 alpha=0.8)
            nx.draw_networkx_edges(G, pos,
                                 edgelist=path_edges,
                                 edge_color=[path_color],
                                 width=2,
                                 alpha=0.8)
            
        plt.title("Chess Opening Network\nHighlighted Victory Paths\n(Color Intensity = Win Rate)", 
                  fontsize=16, pad=20)
        plt.savefig("network_victory_paths.png", dpi=300, bbox_inches='tight')
        plt.close()

# Function to print detailed network analysis
def print_network_analysis(analysis_results):
    print("Degree Centrality (Top 10 Positions):")
    print(sorted(analysis_results['degree_centrality'].items(), key=lambda x: x[1], reverse=True)[:10])

    print("\nBetweenness Centrality (Top 10 Positions):")
    print(sorted(analysis_results['betweenness_centrality'].items(), key=lambda x: x[1], reverse=True)[:10])

    print("\nCommunities (Number of Communities):")
    print(len(analysis_results['communities']))

# Function to analyze winning positions
def analyze_winning_positions(G):
    """Analyze positions and moves that contribute most to winning"""
    print("\n=== Chess Position Analysis Report ===\n")
    
    # 1. Critical Positions Analysis
    print("1. CRITICAL POSITIONS")
    print("--------------------")
    win_rates = []
    for node, data in G.nodes(data=True):
        games = data.get('games', 0)
        if games >= 10:  # Increased minimum games threshold for reliability
            win_rate = data.get('win_rate', 0)
            win_rates.append((node, win_rate, games))
    
    # Sort by win rate and game count
    win_rates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    print("\nA. Most Successful Positions (minimum 10 games):")
    for pos, win_rate, games in win_rates[:5]:
        print(f"\nPosition FEN: {pos.split(' ')[0]}")  # Only show piece placement
        print(f"Games played: {games}")
        print(f"Win rate: {win_rate:.1%}")
        print(f"Statistical significance: {'High' if games >= 20 else 'Medium'}")
    
    # 2. Game-Changing Moves Analysis
    print("\n2. GAME-CHANGING MOVES")
    print("---------------------")
    winning_moves = []
    
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 0)
        if weight >= 5:  # Consider only moves played at least 5 times
            source_wr = G.nodes[u].get('win_rate', 0)
            target_wr = G.nodes[v].get('win_rate', 0)
            win_rate_improvement = target_wr - source_wr
            
            if win_rate_improvement > 0.1:  # 10% improvement threshold
                winning_moves.append({
                    'from_pos': u,
                    'to_pos': v,
                    'improvement': win_rate_improvement,
                    'times_played': weight,
                    'initial_win_rate': source_wr,
                    'final_win_rate': target_wr
                })
    
    # Sort by win rate improvement
    winning_moves.sort(key=lambda x: x['improvement'], reverse=True)
    
    print("\nA. Most Impactful Moves (by win rate improvement):")
    for move in winning_moves[:5]:
        print(f"\nMove Analysis:")
        print(f"Times played: {move['times_played']}")
        print(f"Win rate improvement: {move['improvement']:.1%}")
        print(f"Starting position win rate: {move['initial_win_rate']:.1%}")
        print(f"Resulting position win rate: {move['final_win_rate']:.1%}")
        
    # 3. Strategic Insights
    print("\n3. STRATEGIC INSIGHTS")
    print("-------------------")
    
    # A. Position Type Analysis
    early_game = [wr for pos, wr, games in win_rates if games >= 10 and pos.count(' ') < 20]
    mid_game = [wr for pos, wr, games in win_rates if games >= 10 and 20 <= pos.count(' ') < 40]
    late_game = [wr for pos, wr, games in win_rates if games >= 10 and pos.count(' ') >= 40]
    
    print("\nA. Win Rate Analysis by Game Phase:")
    if early_game:
        print(f"Early game average win rate: {sum(early_game)/len(early_game):.1%}")
    if mid_game:
        print(f"Middle game average win rate: {sum(mid_game)/len(mid_game):.1%}")
    if late_game:
        print(f"Late game average win rate: {sum(late_game)/len(late_game):.1%}")
    
    # B. Critical Move Sequences
    print("\nB. Critical Move Sequences:")
    sequence_threshold = 0.15  # 15% cumulative improvement
    for i, move1 in enumerate(winning_moves[:10]):
        for move2 in winning_moves[i+1:]:
            if move1['to_pos'] == move2['from_pos']:
                total_improvement = move1['improvement'] + move2['improvement']
                if total_improvement > sequence_threshold:
                    print(f"\nStrong move sequence found:")
                    print(f"Total win rate improvement: {total_improvement:.1%}")
                    print(f"Sequence played {min(move1['times_played'], move2['times_played'])} times")
    
    # Add opening classification
    print("\n4. OPENING ANALYSIS")
    print("------------------")
    
    opening_success = defaultdict(lambda: {'wins': 0, 'games': 0})
    for node, data in G.nodes(data=True):
        opening = get_opening_name(node)
        if opening != "Unknown Position":
            opening_success[opening]['games'] += data.get('games', 0)
            opening_success[opening]['wins'] += data.get('games', 0) * data.get('win_rate', 0)
    
    print("\nOpening Success Rates:")
    for opening, stats in opening_success.items():
        if stats['games'] >= 5:  # Minimum games threshold
            win_rate = stats['wins'] / stats['games']
            print(f"\n{opening}:")
            print(f"Games played: {stats['games']}")
            print(f"Win rate: {win_rate:.1%}")

    # Add move sequence naming
    print("\nCritical Move Sequences:")
    for i, move1 in enumerate(winning_moves[:10]):
        for move2 in winning_moves[i+1:]:
            if move1['to_pos'] == move2['from_pos']:
                from_opening = get_opening_name(move1['from_pos'])
                to_opening = get_opening_name(move2['to_pos'])
                print(f"\nSequence: {from_opening} → {to_opening}")
                print(f"Win rate improvement: {move1['improvement'] + move2['improvement']:.1%}")
    
    print("\n=== End of Analysis ===")

# Main function
def main():
    # Load the PGN file with 1000 games
    games = parse_pgn(pgn_file, max_games=1000)  # Using 1000 games

    # Create the move transition network
    print("Creating move network...")
    G, position_count = create_move_network(games, min_position_count=5)  # Reduced min_position_count

    print(f"Network created with {len(G.nodes())} nodes and {len(G.edges())} edges")

    # Analyze the network
    print("Analyzing network...")
    analysis_results = analyze_network(G)

    # Visualize the network
    print("Generating visualizations...")
    visualize_network(G, analysis_results)

    # Print detailed network analysis
    print_network_analysis(analysis_results)

    # Add win rate analysis
    print("\nAnalyzing winning positions and moves...")
    analyze_winning_positions(G)

if __name__ == "__main__":
    print("Starting program...")
    main()