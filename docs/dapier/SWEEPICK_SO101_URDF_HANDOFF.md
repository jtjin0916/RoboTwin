# Handoff the measured SWEEPICK SO-101 URDF

This handoff is **for the SAPIEN simulation SO-101 arm** that was built
from the team's measured replacement base STL. The actual generated
unit-scaled STL and derived URDF are locally generated and **not
tracked by GitHub**. Therefore, asking a teammate to clone the repo
alone does not transfer the working generated mesh.

## On the Ubuntu machine where the current SAPIEN sim works

\`\`\`bash
cd ~/RoboTwin-tabletop
ls -lh \
  assets/embodiments/so101-pgripper/so101_pgripper_sweepick_unit.urdf \
  assets/embodiments/so101-pgripper/assets/Base_SO101_sweepick_unit_m.stl

python scripts/package_sweepick_measured_urdf.py
\`\`\`

Result: \`~/RoboTwin-tabletop/SWEEPICK_SO101_measured_handoff.zip\`.
The packer collects the generated unit-scale URDF, converted STL, all
other STL mesh references from the generated URDF **and** the source
and CuRobo-calibrated URDFs, measured embodiment config, CuRobo config,
and conversion helper script. It rejects any missing dependencies
rather than silently exporting a broken model. It also writes a
\`HANDOFF_README.md\`, \`MANIFEST.txt\`, and \`SHA256SUMS.txt\`.

To check ZIP entries:

\`\`\`bash
unzip -l SWEEPICK_SO101_measured_handoff.zip | head -40
\`\`\`

To copy it to another computer, transfer the ZIP by your team's chosen
means, then extract into a compatible RoboTwin checkout:

\`\`\`bash
unzip SWEEPICK_SO101_measured_handoff.zip -d /path/to/RoboTwin
cd /path/to/RoboTwin
sha256sum -c SHA256SUMS.txt
\`\`\`

If extracting into an existing checkout, review destination paths
and any conflicts first. Do not overwrite teammate custom edits.

## Simulation architecture/limitations

- **SAPIEN:** \`so101_pgripper_sweepick_unit.urdf\`, modified base STL baked
  into metres (NOT the older \`sweepick_measured.urdf\` using \`scale=0.001\`).
- **CuRobo:** the separate \`so101_pgripper_calib.urdf\` referenced by
  measured \`curobo.yml\` remains the planning model.
- **Two arm root transforms and home poses:** measured \`config.yml\`.
- **TurtleBot3 chassis:** separate team-supplied URDF and mesh asset
  referenced via \`SWEEPICK_TURTLEBOT_URDF\`, not part of this archive.
- **OS30A:** measured config currently contains provisional extrinsic;
  team must remeasure actual mounting position before interpreting
  camera-pose randomness as calibrated sensor uncertainty.
- **CuRobo collision limits:** actual TurtleBot chassis and mount
  rails are not fully represented in planner obstacles. Do not
  claim physically collision-safe workspace on this package alone.

For an end-to-end simulation handoff, also clone the correct
RoboTwin Git branch and bring external TurtleBot/ROS2 hardware assets.
