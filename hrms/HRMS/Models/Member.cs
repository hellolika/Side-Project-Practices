using HRMS.Enum;
using HRMS.Exceptions;
using HRMS.Models.Responses;
using Newtonsoft.Json;
using System.Text.RegularExpressions;
using XAct;

namespace HRMS.Models
{
    public class Member : RepositoryBaseResponse
    {
        [JsonProperty("MemberId")]
        public virtual int MemberId { get; set; }

        [JsonProperty("Username")]
        public string Username { get; set; }

        [JsonProperty("Email")]
        public string Email { get; set; }

        [JsonProperty("PhoneNumber")]
        public string PhoneNumber { get; set; }
        
        [JsonProperty("Gender")]
        public string Gender { get; set; }

        [JsonProperty("Password")] public virtual string Password { get; set; }

        [JsonProperty("Address")] public virtual  string Address { get; set; } = string.Empty;

        [JsonProperty("Salary")]
        public decimal Salary { get; set; }

        [JsonProperty("Permission")] public virtual  int Permission { get; set; }

        [JsonProperty("BankAccount")]
        public string BankAccount { get; set; }
        
        [JsonProperty("BankName")]
        public string BankName { get; set; }
        
        [JsonProperty("PositionId")]
        public int PositionId { get; set; }
        
        [JsonProperty("Position")]
        public string Position { get; set; }
        
        [JsonProperty("EmployeeId")]
        public string EmployeeId { get; set; }

        [JsonProperty("IsInProbation")]
        public bool IsInProbation { get; set; }

        [JsonProperty("Remark")] public virtual string Remark { get; set; }

        [JsonProperty("TeamId")]
        public int TeamId { get; set; }  

        [JsonProperty("JobGrade")]
        public int JobGrade { get; set; }
        
        [JsonProperty("JobGradeName")]
        public string JobGradeName { get; set; }

        [JsonProperty("WorkLocationId")] public virtual int WorkLocationId { get; set; }
        
        [JsonProperty("IsSupervisor")]
        public virtual bool IsSupervisor { get; set; }

        [JsonProperty("IsDeleted")]
        public virtual bool IsDeleted { get; set; }

        [JsonProperty("IsFirstTimeUser")]
        public virtual bool? IsFirstTimeUser { get; set; }

        [JsonIgnore] public bool IsResigned { get; set; } = false;

        [JsonProperty("JoinDate")] public DateTimeOffset JoinDate { get; set; } = DateTimeOffset.Now;

        [JsonIgnore] public bool IsAlertProbation { get; set; } = false;
        
        [JsonProperty("Birthday")] public DateTime? Birthday { get; set; }

        [JsonProperty("NationalId")] public string NationalId { get; set; }
        
        [JsonProperty("VehicleType")] public string VehicleType { get; set; }
        
        [JsonProperty("VehicleNumber")] public string VehicleNumber { get; set; }
        
        [JsonProperty("DepartmentId")] public int DepartmentId { get; set; } = 1;
        
        [JsonProperty("DepartmentName")] public virtual string DepartmentName { get; set; }
        
        
        [JsonProperty("IsManager")] public bool IsManager { get; set; }
        
        [JsonProperty("IsCanSeeMemberSalary")] public bool IsCanSeeMemberSalary { get; set; }

        private readonly Regex _emailRegex = new(@"^([\w\.\-]+)@([\w\-]+)((\.(\w){2,3})+)$");

        protected void CheckCommonModels()
        {
            if (string.IsNullOrWhiteSpace(Email))
            {
                ThrowInvalidModelException("Email is required");
            } 
            if (string.IsNullOrWhiteSpace(Gender))
            {
                ThrowInvalidModelException("Gender is required");
            }
            if (!_emailRegex.IsMatch(Email))
            {
                ThrowInvalidModelException("Invalid email format");
            }
            if (string.IsNullOrWhiteSpace(Username))
            {
                ThrowInvalidModelException("Username is required");
            }
            if (decimal.Compare(Salary, 0) < 0)
            {
                ThrowInvalidModelException("Salary cannot be less than 0");
            }
            if (string.IsNullOrWhiteSpace(BankAccount))
            {
                ThrowInvalidModelException("Bank account is required");
            }
            if (TeamId <= 0)
            {
                ThrowInvalidModelException("Team Id is invalid");
            }
            if (WorkLocationId <= 0)
            {
                ThrowInvalidModelException("Work location Id is invalid");
            }            
            if (string.IsNullOrWhiteSpace(BankName))
            {
                ThrowInvalidModelException("Bank Name is required");
            }           
            if (string.IsNullOrWhiteSpace(Position))
            {
                ThrowInvalidModelException("Position is required");
            }            
            if (string.IsNullOrWhiteSpace(EmployeeId))
            {
                ThrowInvalidModelException("Employee Id is required");
            }
        }

        protected static void ThrowInvalidModelException(string message)
        {
            throw new ApiException(ApiErrorEnum.InvalidModelState, message);
        }
    }
}
