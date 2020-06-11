import React, { Component } from "react";
import Button from "./button";
import WellcomeImage from "./wellcome_image";

class Choice extends Component {
  render() {
    let imagesToRender;
    if (this.props.image_ids) {
      imagesToRender = this.props.image_ids.map((id) => {
        return <WellcomeImage id={id}></WellcomeImage>;
      });
    }
    return (
      <div className="pa3 w-50 dib">
        <Button
          name={this.props.choice_name}
          index={this.props.index}
          alt_index={this.props.alt_index}
        ></Button>
        <div>{imagesToRender}</div>
      </div>
    );
  }
}

export default Choice;
