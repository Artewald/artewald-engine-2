use regex::Regex;

fn main() {
    let toml_file = std::fs::read_to_string("Cargo.toml").unwrap();

    let ash_line = toml_file.lines().find(|line| line.contains("ash = ")).unwrap(); 
    let version = match ash_line.split(" = ").count() {
        1 => panic!("No version specified for ash"),
        2 => ash_line.split(" = ").nth(1).unwrap().replace("\"", ""),
        _ => {
            let regex = Regex::new("version = \"([^\"]+)\"").unwrap();
            regex.captures(ash_line).unwrap().get(1).unwrap().as_str().to_string()
        }
    };
    
}
