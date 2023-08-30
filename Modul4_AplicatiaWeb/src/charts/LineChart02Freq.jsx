import React, { useRef, useEffect, useState } from 'react';

import {
  Chart, LineController, LineElement, Filler, PointElement, LinearScale, TimeScale, Tooltip,
} from 'chart.js';
import 'chartjs-adapter-moment';

// Import utilities
import { tailwindConfig, formatValue } from '../utils/Utils';

Chart.register(LineController, LineElement, Filler, PointElement, LinearScale, TimeScale, Tooltip);

function LineChart02({
  data,
  width,
  height
}) {
  
  const canvas = useRef(null);
  const legend = useRef(null);

  useEffect(() => {
    const ctx = canvas.current;
    // eslint-disable-next-line no-unused-vars
    const chart = new Chart(ctx, {
      type: 'line',
      data: data,
      options: {
        chartArea: {
          backgroundColor: tailwindConfig().theme.colors.slate[50],
        },
        layout: {
          padding: 20,
        },
        scales: {
          y: {
            title:{
              display:false,
              text:"% Mulțumire"
            },
            border: {
              display: true,
            },
            grid: {
              beginAtZero: true,
            },
            ticks: {
              maxTicksLimit: 11,
              stepSize: 1,
            },
            suggestedMin:0,
            suggestedMax:1,
            
          },
          x: {
            type: 'time',
            time: {
              parser: 'YYYY-MM-DD',
              unit: 'month',
              displayFormats: {
                month: 'MMM YY',
              },
            },
            border: {
              display: true,
            },
            grid: {
              display: false,
            },
            ticks: {
              autoSkipPadding: 6,
              maxRotation: 0,
              callback: function(value, index, values) {
                const date = new Date(value);
                return date.toLocaleString('ro-RO', { month: 'short', year: 'numeric' });
              },
            },
          },
        },
        plugins: {
          legend: {
            display: false,
          },
          tooltip: {
            callbacks: {
              title: () => false, // Disable tooltip title
              label: (context) => "Nr. recenzii: "+ context.parsed.y,
              afterLabel: (context) => "Data: " + new Date(context.parsed.x).toLocaleString('ro-RO', { month: 'long' }) +' '+new Date(context.parsed.x).getFullYear(),
            },
          },
        },
        interaction: {
          intersect: false,
          mode: 'nearest',
        },
        maintainAspectRatio: false,
        resizeDelay: 200,
      },
      plugins: [
        {
          id: 'htmlLegend',
          afterUpdate(c, args, options) {
            const ul = legend.current;
            if (!ul) return;
            // Remove old legend items
            while (ul.firstChild) {
              ul.firstChild.remove();
            }
            // Reuse the built-in legendItems generator
            const items = c.options.plugins.legend.labels.generateLabels(c);
            items.slice(0, 2).forEach((item) => {
              const li = document.createElement('li');
              li.style.marginLeft = tailwindConfig().theme.margin[3];
              // Button element
              const button = document.createElement('button');
              button.style.display = 'inline-flex';
              button.style.alignItems = 'center';
              button.style.opacity = item.hidden ? '.3' : '';
              button.onclick = () => {
                c.setDatasetVisibility(item.datasetIndex, !c.isDatasetVisible(item.datasetIndex));
                c.update();
              };
              // Color box
              const box = document.createElement('span');
              box.style.display = 'block';
              box.style.width = tailwindConfig().theme.width[3];
              box.style.height = tailwindConfig().theme.height[3];
              box.style.borderRadius = tailwindConfig().theme.borderRadius.full;
              box.style.marginRight = tailwindConfig().theme.margin[2];
              box.style.borderWidth = '3px';
              box.style.borderColor = c.data.datasets[item.datasetIndex].borderColor;
              box.style.pointerEvents = 'none';
              // Label
              const label = document.createElement('span');
              label.style.color = tailwindConfig().theme.colors.slate[500];
              label.style.fontSize = tailwindConfig().theme.fontSize.sm[0];
              label.style.lineHeight = tailwindConfig().theme.fontSize.sm[1].lineHeight;
              const labelText = document.createTextNode(item.text);
              label.appendChild(labelText);
              li.appendChild(button);
              button.appendChild(box);
              button.appendChild(label);
              ul.appendChild(li);
            });
          },
          afterDraw: function(chart, easing) {
            // check if the data is empty
            if (!chart.data.datasets || chart.data.datasets[0].data.length === 0) {
              const ctx = chart.ctx;
              ctx.save();
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.font = 'Inter text-xs font-semibold text-slate-400';
              ctx.fillText('Nu există suficiente date disponibile!', chart.width / 2, chart.height / 2);
              ctx.restore();
            }
          }},
      ],
    });
    return () => chart.destroy();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  return (
    <React.Fragment>
      <div className="px-5 pt-3">
        <div className="flex flex-wrap justify-between items-end">
          {/* <div className="flex items-start">
          </div> */}
            {/* <div className="text-3xl font-bold text-slate-800 mr-2">$1,482</div> */}
            {/* <div className="text-sm font-semibold text-white px-1.5 bg-yellow-500 rounded-full">-22%</div> */}
          
          <div className="grow ml-2">
            <ul ref={legend} className="flex flex-wrap justify-end"></ul>
          </div>
        </div>
      </div>
      {/* Chart built with Chart.js 3 */}
      <div className="grow mr-5">
        <canvas ref={canvas} width={width} height={height}></canvas>
      </div>
    </React.Fragment>
  );
}

export default LineChart02;