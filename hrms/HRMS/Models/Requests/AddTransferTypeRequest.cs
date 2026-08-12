using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
   public class AddTransferTypeRequest
   {
      [JsonProperty("TransferName")]
      public string TransferName { get; set; }
   
      [JsonProperty("IsEnable")]
      public bool IsEnable { get; set; }
   
      [JsonIgnore]
      public int CreateBy { get; set; }
   
      [JsonIgnore]
      public int ModifiedBy { get; set; }
   }
}