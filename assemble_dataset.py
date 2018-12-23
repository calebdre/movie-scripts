from tvdb_api import Tvdb
import requests
from lxml.etree import HTML
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def get_script(url):
    html = requests.get(url).content
    tree = HTML(html)

    try:
        script_html = tree.cssselect(".scrolling-script-container")[0]
        script = "".join([text for text in script_html.itertext()])
        return script
    except Exception as e:
        return None

def get_script_url(show, season, episode):
    return "https://www.springfieldspringfield.co.uk/view_episode_scripts.php?tv-show={}&episode=s{:02d}e{:02d}".format(show, season, episode)

def get_data(packed_data):
    episode_num, query_name, episode, season = packed_data
    url = get_script_url(query_name, season, episode_num)
    script = get_script(url)
    if script is None:
        #tqdm.write("\nSkipping {} s{:02d}e{:02d}\n".format(query_name, season, episode_num))
        return None

    datum = {
        "script": script,
        "rating": episode["siteRating"],
        "name": query_name,
        "overview": episode["overview"],
        "season": season,
        "episode": episode_num,
    }

    return datum

def main(shows):
    tvdb = Tvdb()
    data = []
    pool = Pool()
    for name in tqdm(shows, desc = "Shows", unit = "show"):
        show = tvdb[name]
        
        if name == "Black-ish":
            name += " 2014"
        elif name == "The Good Place":
            name += " 2016"
        elif name == "What I Like About You":
            name += " 2002"
        elif name == "Unbreakable Kimmy Schmidt":
            name += " 2015"
        elif name == "Love":
            name += " 2016"
        elif name == "Insecure":
            name += " 2016"
        elif name == "Jane the Virgin":
            name += " 2014"
        elif name == "The Golden Girls":
            name += " 1985"
        
        num_seasons = len(show.keys())
        num_seasons = min(num_seasons, 25)
        query_name = name.lower().replace("!", "").replace("-", " ").replace(" ", "-").replace("'", "")
        for season in tqdm(range(1, num_seasons), desc = name, unit = "season"):
            num_episodes = len(show[season])
            info = []
            for i in range(1, num_episodes + 1):
                try:
                    info.append((i, query_name, show[season][i], season))
                except Exception as e:
                    #print(e)
                    pass
            for result in tqdm(pool.imap(get_data, info), desc = "Episodes", unit = "episode", total = num_episodes):
                if result is not None:
                    data.append(result)
    pool.close()
    pool.join()

    with open("scripts.csv", "a") as f:
        pd.DataFrame(data).to_csv(f, index = False, sep = "\t", header = False)

if __name__ == "__main__":
    #shows = ["Parks and Recreation", "The Good Place", "Seinfeld", "It's Always Sunny in Philadelphia", "Glee", "Dads", "What I Like About You", "Sabrina the Teenage Witch", "The Office US", "Sex and the City", "Two and a Half Men", "Big Bang Theory", "Boy Meets World", "Roseanne", "The Simpsons", "Will and Grace", "Black-ish", "King of the Hill", "Spongebob Squarepants", "Sherlock", "Rick and Morty"]
    shows = ["All in the Family","Everybody Hates Chris", "Shake It Up!", "New Girl", "Unbreakable Kimmy Schmidt", "Love", "Insecure", "Jane the Virgin", "The Golden Girls", "The Fresh Prince of Bel-Air", "Reno 911!", "Malcolm in the Middle"]
    main(shows)
