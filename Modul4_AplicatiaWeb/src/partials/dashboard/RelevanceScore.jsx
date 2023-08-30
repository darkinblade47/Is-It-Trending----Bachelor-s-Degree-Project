import { React, useEffect, useState } from 'react';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';
import Info from '../../utils/Info';

const RelevanceScore = ({ score }) => {

  const [statusTrending, useStatus] = useState("");

  useEffect(() => {
    if (score >= 0 && score < 2.5) {
      useStatus("Clienții nu s-au declarat multumiți de produsul selectat.")
    }
    else if (score >= 2.5 && score < 3.5) {
      useStatus("Un număr scăzut de clienți s-au declarat mulțumiți de produsul selectat.")
    }
    else if (score >= 3.5 && score < 4.0) {
      useStatus("Un număr moderat de clienți s-au declarat mulțumiți de produsul selectat.")
    }
    else if (score >= 4.0) {
      useStatus("Un număr ridicat de clienți s-au declarat foarte mulțumiți de acest produs.")
    }
  }, [score])

  const smallerFontStyles = {
    fontSize: '12px', // Customize the font size for the smaller font
    fill: '#4f46e5' // Customize the color of the smaller font
    
  };

  const textStyles = {
    fontSize: '24px', // Customize the font size for the default text
    fill: '#4f46e5' // Customize the color of the default text
  };

  return (
    <>
      <div className="flex flex-col col-span-full sm:col-span-3 h-full bg-white shadow-lg rounded-sm border border-slate-200 max-lg:h-72 max-lg:w-full">
        <header className="px-5 py-4 border-b border-slate-100 flex items-center">
          <h2 className="font-semibold text-slate-800">Indicele de relevanță</h2>
          <Info className="ml-2" containerClassName="min-w-44">
            <div className="text-sm">Gradul de mulțumire al clienților în ultimele 6 luni.</div>
          </Info>
        </header>
        <div className='flex flex-col h-full justify-center w-3/4 m-auto max-lg:h-4/5 max-lg:mt-2'>
          <CircularProgressbar value={Math.ceil(score * 100) / 100} maxValue={5} minValue={0} text={ <tspan>
            <tspan style={textStyles}>{Math.ceil(score * 100) / 100}</tspan>
            <tspan style={smallerFontStyles}>/5</tspan>
          </tspan>}
            styles={buildStyles({
              pathColor: `#4f46e5`,
            })
            } />
          <h3 className="font-semibold text-center text-slate-800 m-4 mr-4 ml-4 max-lg:w-full">{statusTrending}</h3>
        </div>

      </div>
    </>
  );
};

export default RelevanceScore;