CREATE TABLE [dbo].[Attendances]
(
	[Id] INT NOT NULL PRIMARY KEY IDENTITY(1, 1), 
    [MemberId] INT NOT NULL, 
    [WorkDate] DATE NOT NULL, 
    [ClockIn] TIME(0) NULL, 
    [ClockOut] TIME(0) NULL, 
    [ClockInLocation] NVARCHAR(500) NULL,
    [ClockOutLocation] NVARCHAR(500) NULL,
    [ClockInRemark] NVARCHAR(500) NULL,
    [ClockOutRemark] NVARCHAR(500) NULL
)
