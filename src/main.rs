#[macro_use]
extern crate clap;
extern crate collapse;
extern crate image;
#[macro_use]
extern crate lazy_static;
extern crate pbr;
extern crate regex;


use std::str::FromStr;


use image::{GenericImage, ImageBuffer};
use clap::{App, AppSettings, Arg, ArgGroup, SubCommand};
use collapse::*;
use pbr::ProgressBar;
use regex::Regex;


lazy_static! {
    static ref MATCH_RANGE: Regex = Regex::new(r"^(-?\d+)..(-?\d+)$").unwrap();
}


fn validate_constraint(constraint: String) -> Result<(), String> {
    let split: Vec<_> = constraint.split(',').collect();

    if split.len() != 4 {
        return Err(format!("Expected four values to the constraint - the x and y of the \
                                 pixel in the input, and the x and y pixels/ranges in the \
                                 output, but got {} values",
                           split.len()));
    } else {
        match (split[0].parse::<i32>(), split[1].parse::<i32>()) {
            (Ok(_), Ok(_)) => (),
            _ => {
                return Err(String::from("Expected the input x and y coordinates to be signed \
                                         integers"));
            }
        }

        match (split[2].parse::<i32>(), split[3].parse::<i32>()) {
            (Err(_), Ok(_)) => {
                if !MATCH_RANGE.is_match(split[2]) {
                    return Err(String::from("Expected the output x coordinate to either be a \
                                             signed integer or a range of signed integers"));
                }
            }
            (Ok(_), Err(_)) => {
                if !MATCH_RANGE.is_match(split[3]) {
                    return Err(String::from("Expected the output y coordinate to either be a \
                                             signed integer or a range of signed integers"));
                }
            }
            (Err(_), Err(_)) => {
                if !MATCH_RANGE.is_match(split[2]) || !MATCH_RANGE.is_match(split[3]) {
                    return Err(String::from("Expected the output x and y coordinates to either \
                                             be a signed integer or a range of signed integers"));
                }
            }
            _ => (),
        }
    }

    Ok(())
}


fn main() {
    let app = App::new("collapse")
        .version(crate_version!())
        .author(crate_authors!())
        .about("A command-line interface for the wavefunction collapse texture synthesis \
                algorithm.")
        .subcommand(SubCommand::with_name("2d")
            .about("The 2D case of the wavefunction collapse algorithm. Samples an input image \
                    and produces an output image.")
            .arg(Arg::with_name("INPUT")
                .help("The input file to sample.")
                .required(true)
                .index(1))
            .arg(Arg::with_name("OUTPUT")
                .help("The output file to generate.")
                .required(true)
                .index(2))
            .arg(Arg::with_name("periodic-input")
                .help("The input periodicity; expects two booleans, corresponding to the x and \
                       y axes. If true, then the input will be processed as wrapping on that \
                       axis. Defaults to false for both axes.")
                .short("p")
                .long("periodic-input")
                .takes_value(true)
                .number_of_values(2)
                .require_delimiter(true)
                .value_name("x, y"))
            .arg(Arg::with_name("periodic-output")
                .help("The output periodicity; expects two booleans, corresponding to the x \
                       and y axes. If true, then the output will be processed as wrapping on \
                       that axis. Defaults to true for both axes.")
                .short("P")
                .long("periodic-output")
                .takes_value(true)
                .number_of_values(2)
                .require_delimiter(true)
                .value_name("x, y"))
            .arg(Arg::with_name("n")
                .help("The sample dimensions; expects two positive nonzero integers, \
                       corresponding to the width and height of the rectangle to be used for \
                       sampling the input. Defaults to 3 for both axes.")
                .short("n")
                .takes_value(true)
                .number_of_values(2)
                .require_delimiter(true)
                .value_name("x, y"))
            .arg(Arg::with_name("no-symmetry")
                .long("no-symmetry")
                .help("Do not augment the sample image with rotations/reflections."))
            .arg(Arg::with_name("all-symmetry")
                .long("all-symmetry")
                .help("Augment the sample image with rotations/reflections, using all members \
                       of the relevant symmetry group. This is the default symmetry setting."))
            .arg(Arg::with_name("identity")
                .help("The original image, since the identity transformation is a no-op. Don't \
                       forget this if you're building up a custom set of symmetries."))
            .arg(Arg::with_name("reflect-x")
                .help("Reflect the x coordinate. Please note that this is not a reflection \
                       over the x axis, but the y axis."))
            .arg(Arg::with_name("reflect-y")
                .help("Reflect the y coordinate. Please note that this is not a reflection \
                       over the y axis, but the x axis."))
            .arg(Arg::with_name("reflect-x-rot90")
                .help("Reflect the x coordinate, and then rotate by 90 degrees. This is \
                       equivalent to a reflection over the line y = -x."))
            .arg(Arg::with_name("reflect-y-rot90")
                .help("Reflect the y coordinate, and then rotate by 90 degrees. This is \
                       equivalent to a reflection over the line y = x."))
            .arg(Arg::with_name("rot90").help("Rotate the image 90 degrees clockwise."))
            .arg(Arg::with_name("rot180").help("Rotate the image 180 degrees clockwise."))
            .arg(Arg::with_name("rot270").help("Rotate the image 270 degrees clockwise."))
            .group(ArgGroup::with_name("symmetry-simple").args(&["no-symmetry", "all-symmetry"]))
            .group(ArgGroup::with_name("symmetry-complex")
                .args(&["identity",
                        "reflect-x",
                        "reflect-y",
                        "reflect-x-rot90",
                        "reflect-y-rot90",
                        "rot90",
                        "rot180",
                        "rot270"])
                .multiple(true)
                .conflicts_with("symmetry-simple"))
            .arg(Arg::with_name("output-dims")
                .help("The dimensions of the output image.")
                .short("d")
                .takes_value(true)
                .number_of_values(2)
                .require_delimiter(true)
                .required(true)
                .value_name("x, y"))
            .arg(Arg::with_name("pixel-constraint")
                .short("c")
                .takes_value(true)
                .multiple(true)
                .value_name("source x, source y, output x (optionally a range x0..x1, x0 \
                             inclusive, x1 exclusive), output y (optionally a range y0..y1, y0 \
                             inclusive, y1 exclusive)")
                .validator(validate_constraint)))
        .settings(&[AppSettings::SubcommandRequiredElseHelp]);

    let matches = app.get_matches();

    if let Some(matches) = matches.subcommand_matches("2d") {
        let periodic_input = if matches.is_present("periodic-input") {
            let vec = values_t!(matches.values_of("periodic-input"), bool)
                .unwrap_or_else(|e| e.exit());
            (vec[0], vec[1])
        } else {
            (false, false)
        };

        let periodic_output = if matches.is_present("periodic-output") {
            let vec = values_t!(matches.values_of("periodic-output"), bool)
                .unwrap_or_else(|e| e.exit());
            (vec[0], vec[1])
        } else {
            (true, true)
        };

        let n = if matches.is_present("n") {
            let vec = values_t!(matches.values_of("n"), usize).unwrap_or_else(|e| e.exit());
            (vec[0], vec[1])
        } else {
            (3, 3)
        };

        let output_dims = {
            let vec = values_t!(matches.values_of("output-dims"), usize)
                .unwrap_or_else(|e| e.exit());
            (vec[0], vec[1])
        };

        let input = matches.value_of("INPUT").unwrap();
        let output = matches.value_of("OUTPUT").unwrap();

        let img = image::open(input).expect("Failed to open input image");

        let symmetries = if matches.is_present("no-symmetry") {
            S2_IDENTITY
        } else if matches.is_present("symmetry-complex") {
            let mut symmetries = Symmetry2::empty();

            if matches.is_present("identity") {
                symmetries |= S2_IDENTITY;
            }

            if matches.is_present("reflect-x") {
                symmetries |= S2_REFLECT_X;
            }

            if matches.is_present("reflect-y") {
                symmetries |= S2_REFLECT_Y;
            }

            if matches.is_present("reflect-x-rot90") {
                symmetries |= S2_REFLECT_X_ROT90;
            }

            if matches.is_present("reflect-y-rot90") {
                symmetries |= S2_REFLECT_Y_ROT90;
            }

            if matches.is_present("rot90") {
                symmetries |= S2_ROTATE_90;
            }

            if matches.is_present("rot180") {
                symmetries |= S2_ROTATE_180;
            }

            if matches.is_present("rot270") {
                symmetries |= S2_ROTATE_270;
            }

            symmetries
        } else {
            Symmetry2::all()
        };

        let input_dims = (img.width(), img.height());

        let constraints: Vec<((usize, usize), (u32, u32))> =
            if matches.is_present("pixel-constraint") {
                matches.values_of("pixel-constraint")
                    .unwrap()
                    .flat_map(|coords| {
                        let coords: Vec<_> = coords.split(',').collect();

                        let (input_x, input_y) = (coords[0].parse::<i32>().unwrap(),
                                                  coords[1].parse::<i32>().unwrap());

                        let input_x = if input_x < 0 {
                            input_x + input_dims.0 as i32
                        } else {
                            input_x
                        } as u32;
                        let input_y = if input_y < 0 {
                            input_y + input_dims.1 as i32
                        } else {
                            input_y
                        } as u32;

                        let outputs_x: Vec<_> =
                            if let Ok(output_x) = coords[2].parse::<i32>() {
                                vec![if output_x < 0 {
                                    output_x + output_dims.0 as i32
                                } else {
                                    output_x
                                } as usize]
                            } else if let Some(caps) = MATCH_RANGE.captures(coords[2]) {
                                let (x0, x1) =
                                    (caps.at(1).and_then(|s| i32::from_str(s).ok()).unwrap(),
                                     caps.at(2).and_then(|s| i32::from_str(s).ok()).unwrap());
                                let x0 = if x0 < 0 {
                                    x0 as usize + output_dims.0
                                } else {
                                    x0 as usize
                                };
                                let x1 = if x1 < 0 {
                                    x1 as usize + output_dims.0
                                } else {
                                    x1 as usize
                                };
                                if x0 < x1 { x0..x1 } else { x1..x0 }.collect()
                            } else {
                                unreachable!();
                            };

                        let outputs_y: Vec<_> =
                            if let Ok(output_y) = coords[3].parse::<i32>() {
                                vec![if output_y < 0 {
                                    output_y + output_dims.0 as i32
                                } else {
                                    output_y
                                } as usize]
                            } else if let Some(caps) = MATCH_RANGE.captures(coords[3]) {
                                let (y0, y1) =
                                    (caps.at(1).and_then(|s| i32::from_str(s).ok()).unwrap(),
                                     caps.at(2).and_then(|s| i32::from_str(s).ok()).unwrap());

                                let y0 = if y0 < 0 {
                                    y0 as usize + output_dims.1
                                } else {
                                    y0 as usize
                                };

                                let y1 = if y1 < 0 {
                                    y1 as usize + output_dims.1
                                } else {
                                    y1 as usize
                                };

                                if y0 < y1 { y0..y1 } else { y1..y0 }.collect()
                            } else {
                                unreachable!();
                            };

                        let mut constrained = Vec::new();

                        for output_x in outputs_x {
                            for &output_y in outputs_y.iter() {
                                constrained.push(((output_x, output_y), (input_x, input_y)));
                            }
                        }

                        constrained.into_iter()
                    })
                    .collect()
            } else {
                vec![]
            };

        let src = Source2::from_image_cli(&img, n, periodic_input, symmetries);

        let mut pb = ProgressBar::new(1);
        pb.message("Building Wave object... ");
        pb.inc();

        let mut wave = Wave::new(output_dims, periodic_output, &src)
            .expect("Error constructing wave");

        let mut pb = ProgressBar::new(constraints.len() as u64);
        pb.message("Propagating individual pixel constraints... ");

        for (output_pos, (x, y)) in constraints {
            pb.inc();
            wave = match wave.constrain(output_pos, img.get_pixel(x, y)) {
                Some(wave) => wave,
                None => {
                    println!("Wave contradiction! The wave has failed to collapse due to a \
                              specified individual pixel constraint: output at {:?} == input at \
                              {:?}",
                             output_pos,
                             (x, y));
                    return;
                }
            };
        }

        let count = wave.len();
        let mut pb = ProgressBar::new(count as u64);

        pb.message("Collapsing wave... ");
        pb.tick();

        while !wave.is_collapsed() {
            wave = match wave.observe() {
                Some(wave) => wave,
                None => {
                    println!("Wave contradiction! The wave has failed to collapse, and no output \
                              was generated. Please try again - maybe another seed will be \
                              kinder to you.");
                    return;
                }
            };

            pb.inc();
        }

        println!("The wave has fully collapsed! Saving result...");

        let pixels = wave.resolve();
        let buffer = ImageBuffer::from_fn(pixels.dim().0 as u32,
                                          pixels.dim().1 as u32,
                                          |x, y| pixels[(x as usize, y as usize)]);

        buffer.save(output).expect("Error saving result");
    }
}
